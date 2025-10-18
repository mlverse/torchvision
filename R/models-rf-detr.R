lwdetr <- nn_module(
  "lw-detr",
  initialize = function(backbone,
                        transformer,
                        segmentation_head,
                        num_classes,
                        num_queries,
                        aux_loss = FALSE,
                        group_detr = 1,
                        two_stage = FALSE,
                        lite_refpoint_refine = FALSE,
                        bbox_reparam = FALSE) {
    self$num_queries <- num_queries
    self$transformer <- transformer
    hidden_dim <- transformer$d_model
    self$class_embed <- torch::nn_linear(hidden_dim, num_classes)
    self$bbox_embed <- MLP(hidden_dim, hidden_dim, 4, 3)
    self$segmentation_head <- segmentation_head

    query_dim = 4
    self$refpoint_embed <- torch::nn_embedding(num_queries * group_detr, query_dim)
    self$query_feat <- torch::nn_embedding(num_queries * group_detr, hidden_dim)
    torch::nn_init$constant_(self$refpoint_embed$weight$data, 0)

    self$backbone <- backbone
    self$aux_loss <- aux_loss
    self$group_detr <- group_detr

    # iter update
    self$lite_refpoint_refine <- lite_refpoint_refine
    if (!self$lite_refpoint_refine) {
      self$transformer$decoder$bbox_embed <- self$bbox_embed
    } else {
      self$transformer$decoder$bbox_embed <- NULL
    }
    self$bbox_reparam <- bbox_reparam

    # init prior_prob setting for focal loss
    prior_prob <- 0.01
    bias_value <- -math$log((1 - prior_prob) / prior_prob)
    self$class_embed$bias$data <- torch::torch_ones(num_classes) * bias_value

    # init bbox_mebed
    torch::nn_init$constant_(self$bbox_embed$layers[-1]$weight$data, 0)
    torch::nn_init$constant_(self$bbox_embed$layers[-1]$bias$data, 0)

    # two_stage
    self$two_stage <- two_stage
    if (self$two_stage) {
      self$transformer$enc_out_bbox_embed <- torch::nn_moduleList(rep(copy$deepcopy(self$bbox_embed), seq_len(group_detr)))
      self$transformer$enc_out_class_embed <- torch::nn_moduleList(rep(copy$deepcopy(self$class_embed), seq_len(group_detr)))
    }
    self$.export <- FALSE

  },
  reinitialize_detection_head = function(num_classes) {
    base <- self$class_embed$weight$shape[1]
    num_repeats <- ceil(num_classes / base)
    self$class_embed$weight$data <- self$class_embed$weight$data$`repeat`(num_repeats, 1)
    self$class_embed$weight$data <- self$class_embed$weight$data[1:num_classes]
    self$class_embed$bias$data <- self$class_embed$bias$data$`repeat`(num_repeats)
    self$class_embed$bias$data <- self$class_embed$bias$data[1:num_classes]

    if (self$two_stage) {
      for (enc_out_class_embed in self$transformer$enc_out_class_embed) {
        enc_out_class_embed$weight$data <- enc_out_class_embed$weight$data$`repeat`(num_repeats, 1)
        enc_out_class_embed$weight$data <- enc_out_class_embed$weight$data[1:num_classes]
        enc_out_class_embed$bias$data <- enc_out_class_embed$bias$data$`repeat`(num_repeats)
        enc_out_class_embed$bias$data <- enc_out_class_embed$bias$data[1:num_classes]
      }
    }
  },
  export = function() {
    self$.export <- TRUE
    self$.forward_origin <- self$forward
    self$forward <- self$forward_export
    for (m in self$named_modules()) {
      if (attr(m, "export") && rlang::is_callable(m$export) && attr(m, "_export") && !m$.export) {
        m$export()
      }
    }
  },
  #' The forward expects a NestedTensor, which consists of:
  #'                - samples$tensor: batched images, of shape \[batch_size x 3 x H x W]
  #'                - samples$mask: a binary mask of shape \[batch_size x H x W], containing 1 on padded pixels
  #'
  #'             It returns a dict with the following elements:
  #'                - "pred_logits": the classification logits (including no-object) for all queries.
  #'                                 Shap = \[batch_size x num_queries x num_classes]
  #'                - "pred_boxes": The normalized boxes coordinates for all queries, represented as
  #'                                (center_x, center_y, width, height). These values are normalized in [0, 1],
  #'                                relative to the size of each individual image (disregarding possible padding).
  #'                                See PostProcess for information on how to retrieve the unnormalized bounding box.
  #'                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
  #'                                 dictionnaries containing the two above keys for each decoder layer.
  forward = function(samples, target = NULL) {

    if (inherits(samples, c(list, torch::torch_Tensor))) {
      samples <- nested_tensor_from_tensor_list(samples)
    }
    c(features, poss) %<-% self$backbone(samples)

    srcs <- list()
    masks <- list()
    for (feat in features) {
      src_mask <- feat$decompose()
      srcs$append(src_mask[[1]])
      masks$append(src_mask[[2]])
      stopifnot(!is.null(src_mask[[2]]))
    }
    if (self$training) {
      refpoint_embed_weight <- self$refpoint_embed$weight
      query_feat_weight <- self$query_feat$weight
    } else {
      # only use one group in inference
      refpoint_embed_weight <- self$refpoint_embed$weight[1:self$num_queries]
      query_feat_weight <- self$query_feat$weight[1:self$num_queries]
    }
    c(hs, ref_unsigmoid, hs_enc, ref_enc) %<-% self$transformer(srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

    if (!is.null(hs)) {
      if (self$bbox_reparam) {
        outputs_coord_delta <- self$bbox_embed(hs)
        outputs_coord_cxcy <- outputs_coord_delta[..., 1:2] * ref_unsigmoid[..., 3:N] + ref_unsigmoid[..., 1:2]
        outputs_coord_wh <- outputs_coord_delta[..., 3:N]$exp() * ref_unsigmoid[..., 3:N]
        outputs_coord <- torch::torch_concat(c(outputs_coord_cxcy, outputs_coord_wh), dim = -1)
      } else {
        outputs_coord <- (self$bbox_embed(hs) + ref_unsigmoid)$sigmoid()
      }
      outputs_class <- self$class_embed(hs)

      if (!is.null(self$segmentation_head)) {
        outputs_masks <- self$segmentation_head(features[[0]]$tensors, hs, tail(samples$tensors$shape,2))
      }
      out <- list('pred_logits' =  outputs_class[-1], 'pred_boxes' =  outputs_coord[-1])
      if (!is.null(self$segmentation_head)) {
        out$pred_masks <- outputs_masks[-1]
      }
      if (self$aux_loss) {
        out$aux_outputs <- self._set_aux_loss(
          outputs_class,
          outputs_coord,
          ifelse(!is.null(self$segmentation_head), outputs_masks , NULL )
        )
      }
    }
    if (self$two_stage) {
      group_detr <- ifelse(self$training, self$group_detr ,1)
      hs_enc_list <- hs_enc$chunk(group_detr, dim = 1)
      cls_enc <- list()
      for (g_idx in seq_len(group_detr)) {
        cls_enc_gidx <- self$transformer$enc_out_class_embed[[g_idx]](hs_enc_list[[g_idx]])
        cls_enc$append(cls_enc_gidx)
      }
      cls_enc <- torch::torch_cat(cls_enc, dim = 1)

      if (!is.null(self$segmentation_head)) {
        masks_enc <- self$segmentation_head(features[1]$tensors,
                                            c(hs_enc, ),
                                            tail(samples$tensors$shape,2),
                                            skip_blocks = TRUE)
        masks_enc <- torch::torch_cat(masks_enc, dim = 1)
      }
      if (!is.null(hs)) {
        out$enc_outputs <- list('pred_logits' =  cls_enc, 'pred_boxes' =  ref_enc)
        if (!is.null(self$segmentation_head)) {
          out$enc_outputs$pred_masks <- masks_enc
        }
      } else {
        out <- list('pred_logits' =  cls_enc, 'pred_boxes' =  ref_enc)
        if (!is.null(self$segmentation_head)) {
          out$pred_masks <- masks_enc
        }
      }
    }
    return(out)

  },
  forward_export = function(tensors) {
    c(srcs, trash, poss) %<-% self$backbone(tensors)
    # only use one group in inference
    refpoint_embed_weight <- self$refpoint_embed$weight[1:self$num_queries]
    query_feat_weight <- self$query_feat$weight[1:self$num_queries]

    c(hs, ref_unsigmoid, hs_enc, ref_enc) %<-% self$transformer(srcs, NULL, poss, refpoint_embed_weight, query_feat_weight)

    outputs_masks <- NULL

    if (!is.null(hs)) {
      if (self$bbox_reparam) {
        outputs_coord_delta <- self$bbox_embed(hs)
        outputs_coord_cxcy <- outputs_coord_delta[.., 1:2] * ref_unsigmoid[.., 3:N] + ref_unsigmoid[.., 1:2]
        outputs_coord_wh <- outputs_coord_delta[.., 3:N]$exp() * ref_unsigmoid[.., 3:N]
        outputs_coord <- torch::torch_concat(c(outputs_coord_cxcy, outputs_coord_wh), dim = -1)
      } else {
        outputs_coord <- (self$bbox_embed(hs) + ref_unsigmoid)$sigmoid()
      }
      outputs_class <- self$class_embed(hs)
      if (!is.null(self$segmentation_head)) {
        outputs_masks <- self$segmentation_head(srcs[[1]], hs, tail(tensors$shape,2))[[1]]
      }
    } else {
      stopifnot("if not using decoder, two_stage must be TRUE" = self$two_stage == TRUE)
      outputs_class <- self$transformer$enc_out_class_embed[[1]](hs_enc)
      outputs_coord <- ref_enc
      if (!is.null(self$segmentation_head)) {
        outputs_masks <- self$segmentation_head(srcs[[1]],
                                                hs_enc,
                                                tail(tensors$shape,2),
                                                skip_blocks = TRUE)[[1]]
      }
    }
    return(list(outputs_coord, outputs_class, outputs_masks))

  },
  .set_aux_loss = function(outputs_class, outputs_coord, outputs_masks) {
    # this is a workaround to make torchscript happy, as torchscript
    # doesn't support dictionary with non-homogeneous values, such
    # as a dict having both a Tensor and a list.
    if (!is.null(outputs_masks)) {
      result <- lapply(seq_along(length(outputs_class) - 1), function(i) {
        list(
          'pred_logits' = outputs_class[[i]],
          'pred_boxes' = outputs_coord[[i]],
          'pred_masks' = outputs_masks[[i]]
        )
      })

    } else {
      # Iterate over the indices and create a list of lists with two elements
      result <- lapply(seq_along(length(outputs_class) - 1), function(i) {
        list('pred_logits' = outputs_class[[i]], 'pred_boxes' = outputs_coord[[i]])
      })
    }

    return(result)
  },
  update_drop_path = function(drop_path_rate,
                              vit_encoder_num_layers) {
    dp_rates <- sapply(torch::torch_linspace(0, drop_path_rate, vit_encoder_num_layers),
                       x$item())
    for (i in seq_len(vit_encoder_num_layers)) {
      if (attr(self$backbone[1]$encoder, 'blocks')) {
        # Not aimv2
        if (attr(self$backbone[1]$encoder$blocks[i]$drop_path, 'drop_prob')) {
          self$backbone[1]$encoder$blocks[i]$drop_path$drop_prob <- dp_rates[i]
        }
      } else {
        # aimv2
        if (attr(self$backbone[1]$encoder$trunk$blocks[i]$drop_path, 'drop_prob')) {
          self$backbone[1]$encoder$trunk$blocks[i]$drop_path$drop_prob <- dp_rates[i]
        }
      }
    }
  },
  update_dropout = function(drop_rate) {
    for (module in self$transformer$modules()) {
      if (inherits(module, torch::nn_dropout)) {
        module$p <- drop_rate
      }
    }
  }
)

#' Create the RF-DETR criterion
#'
#' @param num_classes number of object categories, omitting the special no-object category
#' @param matcher module able to compute a matching between targets and proposals
#' @param weight_dict dict containing as key the names of the losses and as values their relative weight.
#' @param focal_alpha alpha in Focal Loss
#' @param losses  list of all the losses to be applied. See get_loss for list of available losses.
#' @param group_detr Number of groups to speed detr training. Default is 1.
#' @param sum_group_losses: whether or not to sum the losses
#' @param use_varifocal_loss: whether or not to use varifocal loss
#' @param use_position_supervised_loss: whether or not to use positional supervsed loss
#' @param ia_bce_loss: whether or not to use ia binary cros entropy loss
#' @param mask_point_sample_ratio value of the mask-point sample ratio
#' @noRd
set_criterion <- nn_module(
  "set-criterion",
  initialize = function(num_classes,
                        matcher,
                        weight_dict,
                        focal_alpha,
                        losses,
                        group_detr = 1,
                        sum_group_losses = FALSE,
                        use_varifocal_loss = FALSE,
                        use_position_supervised_loss = FALSE,
                        ia_bce_loss = FALSE,
                        mask_point_sample_ratio = 16L) {
    self$num_classes <- num_classes
    self$matcher <- matcher
    self$weight_dict <- weight_dict
    self$losses <- losses
    self$focal_alpha <- focal_alpha
    self$group_detr <- group_detr
    self$sum_group_losses <- sum_group_losses
    self$use_varifocal_loss <- use_varifocal_loss
    self$use_position_supervised_loss <- use_position_supervised_loss
    self$ia_bce_loss <- ia_bce_loss
    self$mask_point_sample_ratio <- mask_point_sample_ratio

  },
  #' Classification loss (Binary focal loss)
  #' @param targets a named list that must include a "labels" named tensor of dim \[nb_target_boxes]
  loss_labels = function(self,
                         outputs,
                         targets,
                         indices,
                         num_boxes,
                         log = TRUE) {
    stopifnot("pred_logits" %in% names(outputs))
    src_logits <- outputs$pred_logits

    idx <- self$.get_src_permutation_idx(indices)
    target_classes_o <- torch::torch_cat(mapply(
      FUN = function(t, index_pair) {
        J <- index_pair[[2]]
        t[["labels"]][J]
      },
      t = targets,
      index_pair = indices,
      simplify = FALSE
    ))

    if (self$ia_bce_loss) {
      alpha <- self$focal_alpha
      gamma <- 2
      src_boxes <- outputs$pred_boxes[idx]
      target_boxes <- torch::torch_cat(mapply(
        FUN = function(t, index_pair) {
          i <- index_pair[[2]]
          t[["boxes"]][i]
        },
        t = targets,
        index_pair = indices,
        simplify = FALSE
      ),
      dim = 0)

      iou_targets <- torch::torch_diag(box_ops$box_iou(
        box_ops$box_cxcywh_to_xyxy(src_boxes$detach()),
        box_ops$box_cxcywh_to_xyxy(target_boxes)
      )[1])
      pos_ious <- iou_targets$clone()$detach()
      prob <- src_logits$sigmoid()
      #init positive weights and negative weights
      pos_weights <- torch::torch_zeros_like(src_logits)
      neg_weights <-  prob^gamma

      idx$append(target_classes_o)

      t <- prob[idx]$pow(alpha) * pos_ious$pow(1 - alpha)
      t <- torch::torch_clamp(t, 0.01)$detach()

      pos_weights[idx] <- t$to(pos_weights$dtype)
      neg_weights[idx] <- 1 - t$to(neg_weights$dtype)
      # a reformulation of the standard loss_ce <- - pos_weights * prob$log() - neg_weights * (1 - prob)$log()
      # with a focus on statistical stability by using fused logsigmoid
      loss_ce <- neg_weights * src_logits - nnf_logsigmoid(src_logits) * (pos_weights + neg_weights)
      loss_ce <- loss_ce$sum() / num_boxes

    } else if (self$use_position_supervised_loss) {
      src_boxes <- outputs$pred_boxes[idx]
      target_boxes <- torch::torch_cat(mapply(
        FUN = function(t, index_pair) {
          i <- index_pair[[2]]
          t[["boxes"]][i]
        },
        t = targets,
        index_pair = indices,
        simplify = FALSE
      ),
      dim = 0)

      iou_targets <- torch::torch_diag(box_ops$box_iou(
        box_ops$box_cxcywh_to_xyxy(src_boxes$detach()),
        box_ops$box_cxcywh_to_xyxy(target_boxes)
      )[1])
      pos_ious <- iou_targets$clone()$detach()
      # pos_ious_func <- pos_ious ^ 2
      pos_ious_func <- pos_ious

      cls_iou_func_targets <- torch::torch_zeros(
        c(src_logits$shape[1], src_logits$shape[2], self$num_classes),
        dtype = src_logits$dtype,
        device = src_logits$device
      )

      idx$append(target_classes_o)
      cls_iou_func_targets[idx] <- pos_ious_func
      norm_cls_iou_func_targets <- cls_iou_func_targets / (
        cls_iou_func_targets$view(cls_iou_func_targets$shape[1], -1, 1)$amax(1, TRUE) + 1e-8
      )
      loss_ce <- position_supervised_loss(
        src_logits,
        norm_cls_iou_func_targets,
        num_boxes,
        alpha = self$focal_alpha,
        gamma = 2
      ) * src_logits$shape[2]

    } else if (self$use_varifocal_loss) {
      src_boxes <- outputs$pred_boxes[idx]
      target_boxes <- torch::torch_cat(mapply(
        FUN = function(t, index_pair) {
          i <- index_pair[[2]]
          t[["boxes"]][i]
        },
        t = targets,
        index_pair = indices,
        simplify = FALSE
      ), dim = 0)

    iou_targets <- torch::torch_diag(box_ops$box_iou(
      box_ops$box_cxcywh_to_xyxy(src_boxes$detach()),
      box_ops$box_cxcywh_to_xyxy(target_boxes)
    )[1])
    pos_ious <- iou_targets$clone()$detach()

    cls_iou_targets <- torch::torch_zeros(
      c(src_logits$shape[1], src_logits$shape[2], self$num_classes),
      dtype = src_logits$dtype,
      device = src_logits$device
    )

    idx$append(target_classes_o)
    cls_iou_targets[idx] <- pos_ious
    loss_ce <- sigmoid_varifocal_loss(
      src_logits,
      cls_iou_targets,
      num_boxes,
      alpha = self$focal_alpha,
      gamma = 2
    ) * src_logits$shape[2]
        } else {
          target_classes <- torch::torch_full(
            src_logits$shape[1:2],
            self$num_classes,
            dtype  =  torch::torch_int64,
            device  =  src_logits$device
          )
          target_classes[idx] <- target_classes_o

          target_classes_onehot <- torch::torch_zeros(
            c(
              src_logits$shape[1],
              src_logits$shape[2],
              src_logits$shape[3]  +  1
            ),
            dtype  =  src_logits$dtype,
            layout  =  src_logits$layout,
            device  =  src_logits$device
          )
          target_classes_onehot$scatter_(2, target_classes$unsqueeze(-1), 1)

          target_classes_onehot <- target_classes_onehot[.., 1:(N  -  1)]
          loss_ce <- sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha  =  self$focal_alpha,
            gamma  =  2
          ) * src_logits$shape[2]
        }
    losses <- list(loss_ce  =  loss_ce)

    if (log) {
      # TODO this should probably be a separate loss, not hacked in this one here
      losses$class_error <- 100 - accuracy(src_logits[idx], target_classes_o)[1]
    }
    return(losses)

  },
  #' Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes
  #' This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
  loss_cardinality = function(self, outputs, targets, indices, num_boxes) {
    torch::torch_no_grad()
    pred_logits <- outputs$pred_logits
    device <- pred_logits$device
    tgt_lengths <- torch::torch_tensor(lapply(targets, function(v)
      length(v$labels)), device  =  device)
    # [len(v$labels) for v in targets], devic = device)
    # Count the number of predictions that are NOT "no-object" (which is the last class)
    card_pred <- (pred_logits$argmax(-1)  = pred_logits$shape[pred_logits$ndim] - 1)$sum(2)
    card_err <- nnf_l1_loss(card_pred$float(), tgt_lengths$float())
    losses <- list(cardinality_error =  card_err)
    return(losses)

  },
  #' Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
  #' targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
  #' The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
  loss_boxes = function(self, outputs, targets, indices, num_boxes) {
    stopifnot("pred_boxes" %in% names(outputs))
    idx <- self$.get_src_permutation_idx(indices)
    src_boxes <- outputs$pred_boxes[idx]
    target_boxes <- torch::torch_cat(mapply(
      FUN = function(t, index_pair) {
        i <- index_pair[[2]]
        t[["boxes"]][i]
      },
      t = targets,
      index_pair = indices,
      simplify = FALSE
    ), dim = 0)

    loss_bbox <- nnf_l1_loss(src_boxes, target_boxes, reduction  =  "none")

    losses <- list()
    losses$loss_bbox <- loss_bbox$sum() / num_boxes

    loss_giou <- 1 - torch::torch_diag(
      box_ops$generalized_box_iou(
        box_ops$box_cxcywh_to_xyxy(src_boxes),
        box_ops$box_cxcywh_to_xyxy(target_boxes)
      )
    )
    losses$loss_giou <- loss_giou$sum() / num_boxes
    return(losses)

  },
  #' Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.
  #' Expects outputs to contain 'pred_masks' of shape c(B, Q, H, W) and targets with key 'masks'.
  loss_masks = function(self, outputs, targets, indices, num_boxes) {
    stopifnot("pred_masks missing in model outputs" = "pred_masks" %in% names(outputs))
    pred_masks <- outputs$pred_masks  # c(B, Q, H, W)
    # gather matched prediction masks
    idx <- self$.get_src_permutation_idx(indices)
    src_masks <- pred_masksc[idx]  # [N, H, W)
    # handle no matches
    if (src_masks$numel() == 0)
    {
      return(list(
        loss_mask_ce =  src_masks$sum(),
        loss_mask_dice = src_masks$sum()
      ))
    }
    # gather matched target masks
    target_masks <- torch::torch_cat(mapply(
      FUN = function(t, index_pair) {
        i <- index_pair[[2]]
        t[["masks"]][i]
      },
      t = targets,
      index_pair = indices,
      simplify = FALSE
    ), dim  =  0)

  # No need to upsample predictions as we are using normalized coordinates :)
  # N x 1 x H x W
  src_masks <- src_masks$unsqueeze(2)
  target_masks <- target_masks$unsqueeze(2)$float()

  num_points <- max(
    src_masks$shape[src_masks$ndim - 1],
    src_masks$shape[src_masks$ndim - 1] * src_masks$shape[src_masks$ndim] %/% self$mask_point_sample_ratio
  )

  torch::with_no_grad({
    # sample point_coords
    point_coords <- get_uncertain_point_coords_with_randomness(src_masks, function(logits)
      calculate_uncertainty(logits), num_points, 3, 0.75, )
    # get gt labels
    point_labels <- point_sample(target_masks,
                                 point_coords,
                                 align_corners = FALSE,
                                 mode = "nearest",)$squeeze(2)
  })

    point_logits <- point_sample(src_masks, point_coords, align_corners =
                                   FALSE, )$squeeze(2)

    losses <- list(
      loss_mask_ce = sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
      loss_mask_dice = dice_loss_jit(point_logits, point_labels, num_boxes),
    )

    rm(src_masks)
    rm(target_masks)
    return(losses)
  },
  .get_src_permutation_idx = function(indices){
    # permute predictions following indices
    batch_idx <- torch::torch_cat(lapply(seq_along(indices), function(i) torch::torch_full_like(indices[[i]][[1]], i)))
    src_idx <- torch::torch_cat(lapply(indices, function(i) i[[1]]))
    return(batch_idx, src_idx)
  },

  .get_tgt_permutation_idx = function(indices){
    # permute targets following indices
    batch_idx <- torch::torch_cat(lapply(seq_along(indices), function(i) torch::torch_full_like(indices[[i]][[2]], i)))
      # c(torch::torch_full_like(tgt, i) for i, (_, tgt) in enumerate(indices)))
    tgt_idx <- torch::torch_cat(lapply(indices, function(i) i[[2]]))
      # c(tgt for (_, tgt) in indices))
    return(batch_idx, tgt_idx)
  },

  get_loss = function(loss, outputs, targets, indices, num_boxes, ...){
    loss_map <- list(
      'labels'=  self$loss_labels,
      'cardinality'=  self$loss_cardinality,
      'boxes'=  self$loss_boxes,
      'masks'=  self$loss_masks
      )
    stopifnot("do you really want to compute {loss} loss?" =  loss %in% loss_map)
    return(loss_map[[!!loss]](outputs, targets, indices, num_boxes, ...))

  },
  #' performs the loss computation.
  #' @param outputs: list of tensors, see the output specification of the model for the format
  #' @param targets: list of lists, such that length(targets) == batch_size.
  #'                 The expected names in each list depends on the losses applied, see each loss' doc
  forward = function(outputs, targets) {
    group_detr <- ifelse(self$training, self$group_detr , 1)
    outputs_without_aux <- outputs$items()
    outputs_without_aux$aux_outputs <- NULL

    # Retrieve the matching between the outputs of the last layer and the targets
    indices <- self$matcher(outputs_without_aux, targets, group_detr=group_detr)

    # Compute the average number of target boxes accross all nodes, for normalization purposes
    num_boxes <- sum(lapply(targets$labels, length))
    if (!self$sum_group_losses) {
      num_boxes <- num_boxes * group_detr
    }
    num_boxes <- torch::torch_as_tensor(num_boxes, dtype=torch::torch_float, device=next(iter(outputs$values()))$device)
    if (is_dist_avail_and_initialized()) {
      torch::torch_distributed$all_reduce(num_boxes)
    }
    num_boxes <- torch::torch_clamp(num_boxes / get_world_size(), min=1)$item()

    # Compute all the requested losses
    losses <- list()
    for (loss in self$losses) {
      losses$update(self$get_loss(loss, outputs, targets, indices, num_boxes))
    }
    # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    if ("aux_outputs" %in% outputs) {
      for (i in seq_along(outputs$aux_outputs)) {
        indices <- self$matcher(outputs$aux_outputs[[i]], targets, group_detr=group_detr)
        for (loss in self$losses){
          kwargs <- list()
          if (loss == "labels") {
            # Logging is enabled only for the last layer
            kwargs$log <- FALSE
          }
          l_dict <- self$get_loss(loss, outputs$aux_outputs[[i]], targets, indices, num_boxes, ...)
          l_dict <- list(glue::glue("{names(l_dict$items())}_{i}" = l_dict$items()))
          losses$update(l_dict)
      }
    }

    if ("enc_outputs" %in% outputs) {
        enc_outputs <- outputs$enc_outputs
        indices <- self$matcher(enc_outputs, targets, group_detr=group_detr)
        for (loss in self$losses) {
          kwargs <- list()
            if (loss == "labels") {
              # Logging is enabled only for the last layer
              kwargs$log <- FALSE
            }
            l_dict <- self$get_loss(loss, enc_outputs, targets, indices, num_boxes, ...)
            l_dict <- list(glue::glue("{names(l_dict$items())}_{i}" = l_dict$items()))
            losses$update(l_dict)
        }
      }
    return(losses)
    }
  }
)

#' Sigmoid Focal loss
#'
#' Loss used in RetinaNet for dense detection: https:%/%arxiv$org/abs/1708.02002.
#'
#' @param inputs: A float tensor of arbitrary shape.
#'             The predictions for each example.
#' @param targets: A float tensor with the same shape as inputs. Stores the binary
#'              classification label for each element in inputs
#'             (0 for the negative class and 1 for the positive class).
#' @param alpha: (optional) Weighting factor in range (0,1) to balance
#'             positive vs negative examples. Default <- -1 (no weighting).
#' @param gamma: Exponent of the modulating factor (1 - p_t) to
#'            balance easy vs hard examples.
#' @return Loss tensor
#' @noRd
#' @importFrom torch nnf_binary_cross_entropy_with_logits
sigmoid_focal_loss = function(inputs, targets, num_boxes, alpha = 0.25, gamma = 2){

  prob <- inputs$sigmoid()
  ce_loss <- nnf_binary_cross_entropy_with_logits(inputs, targets, reduction="none")
  p_t <- prob * targets + (1 - prob) * (1 - targets)
  loss <- ce_loss * ((1 - p_t) ^ gamma)

  if (alpha >= 0) {
    alpha_t <- alpha * targets + (1 - alpha) * (1 - targets)
    loss <- alpha_t * loss
  }
  return(loss$mean(1)$sum() / num_boxes)
}


#' Sigmoid Varifocal loss
#'
#' used in RetinaNet for dense detection: https:%/%arxiv$org/abs/1708.02002.
#'
#' @param inputs: A float tensor of arbitrary shape.
#'             The predictions for each example.
#' @param targets: A float tensor with the same shape as inputs. Stores the binary
#'              classification label for each element in inputs
#'             (0 for the negative class and 1 for the positive class).
#' @param alpha: (optional) Weighting factor in range (0,1) to balance
#'             positive vs negative examples. Default <- -1 (no weighting).
#' @param gamma: Exponent of the modulating factor (1 - p_t) to
#'            balance easy vs hard examples.
#' @return Loss tensor
#' @noRd
sigmoid_varifocal_loss = function(inputs, targets, num_boxes, alpha = 0.25, gamma = 2){
  prob <- inputs$sigmoid()
  focal_weight <- targets * (targets > 0)$float() + (1 - alpha) * (prob - targets)$abs()$pow(gamma) * (targets <= 0)$float()
  ce_loss <- nnf_binary_cross_entropy_with_logits(inputs, targets, reduction="none")
  loss <- ce_loss * focal_weight

  return(loss$mean(1)$sum() / num_boxes)
}


#' Position Supervised loss
#'
#' used in RetinaNet for dense detection: https:%/%arxiv$org/abs/1708.02002.
#'
#' @param inputs: A float tensor of arbitrary shape.
#'             The predictions for each example.
#' @param targets: A float tensor with the same shape as inputs. Stores the binary
#'              classification label for each element in inputs
#'             (0 for the negative class and 1 for the positive class).
#' @param num_boxes: number of bounding box
#' @param alpha: (optional) Weighting factor in range (0,1) to balance
#'             positive vs negative examples. Default <- -1 (no weighting).
#' @param gamma: Exponent of the modulating factor (1 - p_t) to
#'            balance easy vs hard examples.
#' @return Loss tensor
#' @noRd
position_supervised_loss = function(inputs, targets, num_boxes, alpha= 0.25, gamma = 2){
  prob <- inputs$sigmoid()
  ce_loss <- nnf_binary_cross_entropy_with_logits(inputs, targets, reduction="none")
  loss <- ce_loss * (torch::torch_abs(targets - prob) ^ gamma)

  if (alpha >= 0) {
    alpha_t <- alpha * (targets > 0)$float() + (1 - alpha) * (targets <= 0)$float()
    loss <- alpha_t * loss

    return(loss$mean(1)$sum() / num_boxes)
  }
}



#' DICE loss.
#'
#' Compute the DICE loss, similar to generalized IOU for masks
#'
#' @param inputs: A float tensor of arbitrary shape.
#'                The predictions for each example.
#' @param targets: A float tensor with the same shape as inputs. Stores the binary
#'                 classification label for each element in inputs
#'                (0 for the negative class and 1 for the positive class).
#' @noRd
dice_loss = function( inputs, targets, num_masks){

  inputs <- inputs$sigmoid()
  inputs <- inputs$flatten(1)
  numerator <- 2 * (inputs * targets)$sum(-1)
  denominator <- inputs$sum(-1) + targets$sum(-1)
  loss <- 1 - (numerator + 1) / (denominator + 1)
  return(loss$sum() / num_masks)


  dice_loss_jit <- torch::torch_jit$script(
    dice_loss
  )  # type: torch::torch_jit$ScriptModule


}
#' Sigmoid cross-entropy loss
#'
#' @param inputs: A float tensor of arbitrary shape.
#'         The predictions for each example.
#' @param targets: A float tensor with the same shape as inputs. Stores the binary
#'              classification label for each element in inputs
#'          (0 for the negative class and 1 for the positive class).
#' @param num_masks: number of masks
#' @returns Loss tensor
sigmoid_ce_loss = function(inputs,targets, num_masks){
  loss <- nnf_binary_cross_entropy_with_logits(inputs, targets, reduction="none")

  return(loss$mean(1)$sum() / num_masks)


  sigmoid_ce_loss_jit <- torch::torch_jit$script(
    sigmoid_ce_loss
  )  # type: torch::torch_jit$ScriptModule
}


#' Compute Uncertainty
#'
#' We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
#'     foreground class in `classes`.
#'
#' @param logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
#'         class-agnostic, where R is the total number of predicted masks in all images and C is
#'         the number of foreground classes. The values are logits.
#' @return A tensor of shape (R, 1, ...) that contains uncertainty scores with
#'         the most uncertain locations having the highest uncertainty score.
calculate_uncertainty = function(logits){

  stopifnot(logits$shape[1] == 1)
  gt_class_logits <- logits$clone()
  return(-(torch::torch_abs(gt_class_logits)))
}


#' PostProcess to coco API
#'
#' converts the model's output into the format expected by the coco api
#' @noRd
post_process <- torch::nn_module(
  "PostProcess",
  initialize = function(num_select=300) {
    self$num_select <- num_select
  },
  #' Perform the computation
  #'
  #' @param outputs: raw outputs of the model
  #' @param target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
  #'             For evaluation, this must be the original image size (before any data augmentation)
  #'             For visualization, this should be the image size after data augment, but before padding
  forward = function(outputs, target_sizes) {
    torch::with_no_grad({
      out_logits <- outputs$pred_logits
      out_bbox <- outputs$pred_boxes
      out_masks <- outputs$get('pred_masks', NULL)

      stopifnot(length(out_logits) == length(target_sizes))
      stopifnot(target_sizes$shape[2] == 2)

      prob <- out_logits$sigmoid()
      c(topk_values, topk_indexes) %<-% torch::torch_topk(prob$view(out_logits$shape[0], -1), self$num_select, dim=1)
      scores <- topk_values
      topk_boxes <- topk_indexes %/% out_logits$shape[3]
      labels <- topk_indexes %% out_logits$shape[3]
      boxes <- box_convert(out_bbox, "cxcywh", "xyxy")
      boxes <- torch::torch_gather(boxes, 1, topk_boxes$unsqueeze(-1)$`repeat`(c(1,1,4)))

      # and from relative [0, 1] to absolute [0, height] coordinates
      c(img_h, img_w) %<-% target_sizes$unbind(1)
      scale_fct <- torch::torch_stack(c(img_w, img_h, img_w, img_h), dim=2)
      boxes <- boxes * scale_fct[, NULL, ]

      # Optionally gather masks corresponding to the same top-K queries and resize to original size
      results <- list()
      if (!is.null(out_masks)) {
        for (i in range(out_masks$shape[1])) {
          res_i <- list(scores =  scores[i], 'labels'=  labels[i], 'boxes'=  boxes[i])
          k_idx <- topk_boxes[i]
          masks_i <- torch::torch_gather(out_masks[i], 0, k_idx$unsqueeze(-1)$unsqueeze(-1)$`repeat`(c(1, tail(out_masks$shape, 2))))  # c(K, Hm, Wm)
          c(h, w) %<-% target_sizes[i]$tolist()
          masks_i <- nnf_interpolate(masks_i$unsqueeze(2), size=c(h, w), mode='bilinear', align_corners=FALSE)  # [K,1,H,W]
          res_i$masks <- masks_i > 0.0
          results$append(res_i)
        }
      } else {
        results <- list('scores'=  scores, 'labels'=  labels, 'boxes'=  boxes)
      }
    })
    return(results)
  }
)

#' simple MLP
#'
#'  Very simple multi-layer perceptron (also called FFN)
#'  @noRd
mlp <- torch::nn_module(
  "MLP",
  initialize = function(input_dim, hidden_dim, output_dim, num_layers){
    self$num_layers <- num_layers
    h <- rep(hidden_dim, num_layers - 1)
    input_dims <- c(input_dim, h)
    output_dims <- c(h, output_dim)
    layers_list <- list()
    for (i in seq_along(input_dims)) {
      layers_list[[i]] <- nn_linear(input_dims[i], output_dims[i])
    }
    self$layers <- nn_module_list(layers_list)

  },
  forward = function(x){
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      x <- layer(x)
      if (i < self$num_layers) {
        x <- nnf_relu(x)
      }
      return(x)
    }
  }
)

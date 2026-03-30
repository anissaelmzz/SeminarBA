import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GTM import GTM


def compute_forecast_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, erp_epsilon: float = 0.1):
    """
    y_true, y_pred: shape [N, H]
    Returns scalar tensors: wape, mae, ts, erp
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    abs_err = torch.abs(y_true - y_pred)

    # Global MAE and WAPE
    mae = abs_err.mean()
    wape = 100.0 * abs_err.sum() / y_true.sum().clamp(min=1e-12)

    # Per-series MAE: shape [N]
    mae_per_series = abs_err.mean(dim=1).clamp(min=1e-12)

    # Per-series TS, then average
    signed_error_per_series = (y_true - y_pred).sum(dim=1)
    ts_per_series = signed_error_per_series / mae_per_series
    ts = ts_per_series.mean()

    # Per-series ERP, then average
    erp_per_series = (abs_err >= erp_epsilon).float().sum(dim=1)
    erp = erp_per_series.mean()

    return wape, mae, ts, erp


class RetrievalAugmentedGTM(GTM):
    """
    Simple retrieval-augmented GTM baseline.

    Reuses the original GTM encoders, then fuses the static feature
    representation with a projected retrieval summary vector.
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers,
        use_text,
        use_img,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        num_trends,
        gpu_num,
        retrieval_dim=12,
        use_encoder_mask=1,
        autoregressive=False,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_text=use_text,
            use_img=use_img,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            trend_len=trend_len,
            num_trends=num_trends,
            gpu_num=gpu_num,
            use_encoder_mask=use_encoder_mask,
            autoregressive=autoregressive,
        )

        self.retrieval_projection = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.retrieval_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.validation_outputs = []

    def forward(self, category, color, fabric, temporal_features, gtrends, images, retrieval_summary):
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        static_feature_fusion = self.static_feature_encoder(
            img_encoding, text_encoding, dummy_encoding
        )

        retrieval_emb = self.retrieval_projection(retrieval_summary)
        augmented_static = self.retrieval_fusion(
            torch.cat([static_feature_fusion, retrieval_emb], dim=1)
        )

        if self.autoregressive == 1:
            tgt = torch.zeros(
                self.output_len,
                gtrend_encoding.shape[1],
                gtrend_encoding.shape[-1],
                device=augmented_static.device,
            )
            tgt[0] = augmented_static
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory, tgt_mask)
            forecast = self.decoder_fc(decoder_out)
        else:
            tgt = augmented_static.unsqueeze(0)
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory)
            forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def training_step(self, train_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, retrieval_summary = train_batch
        forecasted_sales, _ = self.forward(
            category, color, fabric, temporal_features, gtrends, images, retrieval_summary
        )
        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, val_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, retrieval_summary = val_batch
        forecasted_sales, _ = self.forward(
            category, color, fabric, temporal_features, gtrends, images, retrieval_summary
        )

        self.validation_outputs.append(
            {
                "item_sales": item_sales.detach(),
                "forecasted_sales": forecasted_sales.detach(),
            }
        )

    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return

        item_sales = torch.cat([x["item_sales"] for x in self.validation_outputs], dim=0)
        forecasted_sales = torch.cat([x["forecasted_sales"] for x in self.validation_outputs], dim=0)

        # Rescaled/original-unit versions
        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065

        # Training loss stays on normalized scale
        val_loss = F.mse_loss(item_sales, forecasted_sales)

        # Normalized metrics
        val_wape_norm, val_mae_norm, val_ts_norm, val_erp_norm = compute_forecast_metrics(
            item_sales,
            forecasted_sales,
            erp_epsilon=0.1,
        )

        # Rescaled/original-unit metrics
        val_wape, val_mae, val_ts, val_erp = compute_forecast_metrics(
            rescaled_item_sales,
            rescaled_forecasted_sales,
            erp_epsilon=0.1,
        )

        # Log normalized metrics
        self.log("val_wape_norm", val_wape_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_mae_norm", val_mae_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_ts_norm", val_ts_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_erp_norm", val_erp_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Log rescaled metrics
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_wape", val_wape, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ts", val_ts, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_erp", val_erp, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        print(
            f"Validation normalized | "
            f"MAE: {val_mae_norm.item():.3f} | "
            f"WAPE: {val_wape_norm.item():.3f} | "
            f"TS: {val_ts_norm.item():.3f} | "
            f"ERP: {val_erp_norm.item():.3f}"
        )

        print(
            f"Validation rescaled | "
            f"MAE: {val_mae.item():.3f} | "
            f"WAPE: {val_wape.item():.3f} | "
            f"TS: {val_ts.item():.3f} | "
            f"ERP: {val_erp.item():.3f}"
        )

        self.validation_outputs.clear()
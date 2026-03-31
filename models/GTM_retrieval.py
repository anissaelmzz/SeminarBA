import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GTM import GTM, PositionalEncoding, TimeDistributed


def compute_forecast_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, erp_epsilon: float = 0.1):
    y_true = y_true.float()
    y_pred = y_pred.float()

    abs_err = torch.abs(y_true - y_pred)
    mae = abs_err.mean()
    wape = 100.0 * abs_err.sum() / y_true.sum().clamp(min=1e-12)

    mae_per_series = abs_err.mean(dim=1).clamp(min=1e-12)
    signed_error_per_series = (y_true - y_pred).sum(dim=1)
    ts_per_series = signed_error_per_series / mae_per_series
    ts = ts_per_series.mean()

    erp_per_series = (abs_err >= erp_epsilon).float().sum(dim=1)
    erp = erp_per_series.mean()
    return wape, mae, ts, erp


class RetrievalMemoryEncoder(nn.Module):
    """
    Encode a retrieved 12-week sales curve as a decoder-memory sequence.
    """

    def __init__(self, hidden_dim, seq_len, num_heads=4):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        self.pos_embedding = PositionalEncoding(hidden_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, retrieval_curve):
        # retrieval_curve: [batch, seq_len]
        retrieval_curve = retrieval_curve.unsqueeze(-1)              # [batch, seq_len, 1]
        retrieval_emb = self.input_linear(retrieval_curve)          # [batch, seq_len, hidden_dim]
        retrieval_emb = self.pos_embedding(retrieval_emb.permute(1, 0, 2))  # [seq_len, batch, hidden_dim]
        retrieval_emb = self.encoder(retrieval_emb)
        return retrieval_emb


class RetrievalAugmentedGTM(GTM):
    """
    Retrieval-augmented GTM using decoder-memory injection.

    Retrieved historical sales curves are encoded as an additional memory
    sequence that the decoder can attend to alongside Google Trends.
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

        self.retrieval_encoder = RetrievalMemoryEncoder(
            hidden_dim=hidden_dim,
            seq_len=retrieval_dim,
            num_heads=num_heads,
        )

        self.validation_outputs = []

    def _build_memory(self, gtrend_encoding, retrieval_curve=None, retrieval_available=None):
        """
        Build decoder memory by concatenating:
        - GTM Google Trends encoding
        - retrieval encoding from historical analog sales curve

        Also build a memory_key_padding_mask so that products with no valid
        retrieved neighbors effectively ignore the retrieval memory.
        """
        memory = gtrend_encoding
        memory_key_padding_mask = None

        if retrieval_curve is None or retrieval_available is None:
            return memory, memory_key_padding_mask

        retrieval_encoding = self.retrieval_encoder(retrieval_curve)   # [retrieval_dim, batch, hidden_dim]
        memory = torch.cat([gtrend_encoding, retrieval_encoding], dim=0)

        batch_size = gtrend_encoding.shape[1]

        # Google Trends memory is always available
        gtrend_mask = torch.zeros(
            (batch_size, gtrend_encoding.shape[0]),
            dtype=torch.bool,
            device=gtrend_encoding.device,
        )

        # If retrieval_available == False for an item, mask out all retrieval tokens
        retrieval_mask = (~retrieval_available.bool()).unsqueeze(1).expand(
            -1, retrieval_encoding.shape[0]
        )

        memory_key_padding_mask = torch.cat([gtrend_mask, retrieval_mask], dim=1)
        return memory, memory_key_padding_mask

    def forward(
        self,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        retrieval_curve,
        retrieval_available,
    ):
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        static_feature_fusion = self.static_feature_encoder(
            img_encoding,
            text_encoding,
            dummy_encoding,
        )

        memory, memory_key_padding_mask = self._build_memory(
            gtrend_encoding=gtrend_encoding,
            retrieval_curve=retrieval_curve,
            retrieval_available=retrieval_available,
        )

        if self.autoregressive == 1:
            tgt = torch.zeros(
                self.output_len,
                gtrend_encoding.shape[1],
                gtrend_encoding.shape[-1],
                device=static_feature_fusion.device,
            )
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)

            decoder_out, attn_weights = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            forecast = self.decoder_fc(decoder_out)
        else:
            tgt = static_feature_fusion.unsqueeze(0)

            decoder_out, attn_weights = self.decoder(
                tgt,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def training_step(self, train_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, retrieval_curve, retrieval_available = train_batch

        forecasted_sales, _ = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            retrieval_curve,
            retrieval_available,
        )

        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, val_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, retrieval_curve, retrieval_available = val_batch

        forecasted_sales, _ = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            retrieval_curve,
            retrieval_available,
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

        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065

        val_loss = F.mse_loss(rescaled_item_sales, rescaled_forecasted_sales)

        val_wape_norm, val_mae_norm, val_ts_norm, val_erp_norm = compute_forecast_metrics(
            item_sales,
            forecasted_sales,
            erp_epsilon=0.1,
        )
        val_wape, val_mae, val_ts, val_erp = compute_forecast_metrics(
            rescaled_item_sales,
            rescaled_forecasted_sales,
            erp_epsilon=0.1,
        )

        self.log("val_wape_norm", val_wape_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_mae_norm", val_mae_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_ts_norm", val_ts_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_erp_norm", val_erp_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
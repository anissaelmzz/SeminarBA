import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GTM import GTM


class RetrievalAugmentedGTM(GTM):
    """
    Simple retrieval-augmented GTM baseline.

    We reuse the original GTM encoders, then fuse the static feature
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

    def forward(self, category, color, fabric, temporal_features, gtrends, images, retrieval_summary):
        # Original GTM encodings
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        static_feature_fusion = self.static_feature_encoder(
            img_encoding, text_encoding, dummy_encoding
        )

        # Retrieval augmentation
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
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, retrieval_summary = val_batch
        forecasted_sales, _ = self.forward(
            category, color, fabric, temporal_features, gtrends, images, retrieval_summary
        )
        return item_sales.squeeze(), forecasted_sales.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales, forecasted_sales = [x[0] for x in val_step_outputs], [x[1] for x in val_step_outputs]
        item_sales, forecasted_sales = torch.stack(item_sales), torch.stack(forecasted_sales)

        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065

        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)

        self.log("val_mae", mae)
        self.log("val_loss", loss)

        print("Validation MAE:", mae.detach().cpu().numpy(), "LR:", self.optimizers().param_groups[0]["lr"])
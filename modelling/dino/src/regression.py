import torch
from torch import nn

class ViTForRegression(nn.Module):
        def __init__(self, base_model, emb_size=768, output_size=1):
            super().__init__()
            self.base_model = base_model
            # Assuming the original model outputs 768 features from the transformer
            self.regression_head = nn.Linear(emb_size, output_size)  # Output one continuous variable

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            # We use the last hidden state
            return torch.sigmoid(self.regression_head(outputs))
        
class ViTForRegressionMSCrossAttention(nn.Module):
        def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, output_size=1):
            super().__init__()
            self.base_models = base_models
            self.grouped_bands = torch.tensor(grouped_bands) - 1
            self.cross_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8)
            
            # Update the input size of the regression head to handle concatenation of 4 embeddings
            self.regression_head = nn.Linear(emb_size * len(grouped_bands), output_size)

        def forward(self, pixel_values):
            # Extract outputs from each base model with specific band groups
            outputs = [self.base_models[i](pixel_values[:, self.grouped_bands[i], :, :]) for i in range(len(self.base_models))]
            
            # Stack and permute outputs for multihead attention
            outputs = torch.stack(outputs, dim=0)  # Shape: [num_views, batch_size, emb_size]
            
            # Apply cross-attention
            attn_output, _ = self.cross_attention(outputs, outputs, outputs)  # Shape: [num_views, batch_size, emb_size]
            
            # Concatenate the attention output across all views
            concat_output = torch.cat([attn_output[i] for i in range(attn_output.size(0))], dim=-1)  # Shape: [batch_size, emb_size * num_views]
            
            # Pass through regression head
            return torch.sigmoid(self.regression_head(concat_output))
        
class ViTForRegressionMS(nn.Module):
        def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, output_size=1):
            super().__init__()
            self.base_models = base_models
            self.grouped_bands = torch.tensor(grouped_bands) - 1
            
            # Update the input size of the regression head to handle concatenation of 4 embeddings
            self.regression_head = nn.Linear(emb_size * len(grouped_bands), output_size)

        def forward(self, pixel_values):
            # Extract outputs from each base model with specific band groups
            outputs = [self.base_models[i](pixel_values[:, self.grouped_bands[i], :, :]) for i in range(len(self.base_models))]
            
            # Stack and permute outputs for multihead attention
            outputs = torch.stack(outputs, dim=0)  # Shape: [num_views, batch_size, emb_size]
            
            # Apply cross-attention
            # attn_output, _ = self.cross_attention(outputs, outputs, outputs)  # Shape: [num_views, batch_size, emb_size]
            
            # Concatenate the attention output across all views
            concat_output = torch.cat([outputs[i] for i in range(outputs.size(0))], dim=-1)  # Shape: [batch_size, emb_size * num_views]
            
            # Pass through regression head
            return torch.sigmoid(self.regression_head(concat_output))


class ViTForRegressionMSWithUncertainty(nn.Module):
        def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, predict_target=1):
            super().__init__()
            self.base_models = base_models
            self.grouped_bands = torch.tensor(grouped_bands) - 1
            self.cross_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8)
            
            # Update the regression head to output both mean and uncertainty
            # The output size is doubled to handle both prediction (mean) and log variance
            self.regression_head = nn.Linear(emb_size * len(grouped_bands), predict_target * 2)

        def forward(self, pixel_values):
            # Extract outputs from each base model with specific band groups
            outputs = [self.base_models[i](pixel_values[:, self.grouped_bands[i], :, :]) for i in range(len(self.base_models))]
            
            # Stack and permute outputs for multihead attention
            outputs = torch.stack(outputs, dim=0)  # Shape: [num_views, batch_size, emb_size]
            
            # Apply cross-attention
            attn_output, _ = self.cross_attention(outputs, outputs, outputs)  # Shape: [num_views, batch_size, emb_size]
            
            # Concatenate the attention output across all views
            concat_output = torch.cat([attn_output[i] for i in range(attn_output.size(0))], dim=-1)  # Shape: [batch_size, emb_size * num_views]
            
            # Pass through regression head to get mean and log variance
            regression_output = self.regression_head(concat_output)  # Shape: [batch_size, predict_target * 2]
            
            # Split the output into mean and log variance
            mean, log_var = torch.chunk(regression_output, 2, dim=-1)  # Each is of shape [batch_size, predict_target]
            
            # Calculate variance and uncertainty (variance must be positive, so apply exp)
            variance = torch.exp(log_var)  # Shape: [batch_size, predict_target]
            
            return mean, variance
import torchvision.models as models
import torch.nn as nn
import torch


"""
    @class M2: Used to detect the type of stimming behaviour.
        @component Resnet (or equivalent) base CNN model
        @component Bi-directional LSTM
        @component Multihead attention model
        @component Fully-connected NN

        @param dropout_rate: The dropout rate used by the model
            @note Used by the LSTM block as-is, but 0,25 * dropout_rate is used for the Attention block
            @note A fraction of dropout_rate can be applied to the Fully-connected NN block by the user
        @param dim_embedding: The embedding dimension that pertains to the LSTM block
        @param dim_hidden: The hidden dimension pertaining to the LSTM block
        @param num_lstm_layers: The number of bidirectional LSTM cells
        @param dim_fc_layer_1: The dimension of the first hidden layer 
        @param dim_fc_layer_2: The dimension of the second hidden layer
        @param base_model: The base CNN feature extractor model of the combined model
            @default torchvision.models.resnet18(pretrained = True)
            @note If another model is used, the user must ensure that its last layer should output a vector of length dim_embedding
        @param n_frames: The number of frames of the input video
            @default 40
        @param use_movenet: Flag to select if movenet vectors are considered for extracting the model's output
            @default True
        @param n_classes: Array containing the respective convolutional stride values
            @default 3
"""
class SSBDModel2(nn.Module):
    def __init__(self, dropout_rate, dim_embedding, dim_hidden, num_lstm_layers,
                 dim_fc_layer_1, dim_fc_layer_2,
                 base_model = models.resnet18(pretrained = True),
                 n_frames = 40,
                 use_movenet = True,
                 n_classes = 3):
        super(SSBDModel2, self).__init__()
        
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden
        self.num_lstm_layers = num_lstm_layers
        self.dim_fc_layer_1 = dim_fc_layer_1
        self.dim_fc_layer_2 = dim_fc_layer_2
        self.base_model = base_model
        self.n_frames = n_frames
        self.use_movenet = use_movenet

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        """
            @block Resnet architecture (base model)
        """
        self.base_model.fc = nn.Linear(
            self.base_model.fc.in_features, 
            self.dim_embedding
        )
        self.base_model.fc.requires_grad = True
        
        """
            @block Bidirectional LSTM
        """
        self.lstm_layer = nn.LSTM(
            self.dim_embedding,
            self.dim_hidden,
            num_layers = self.num_lstm_layers,
            dropout = self.dropout_rate,
            bidirectional = True
        )
        
        """
            @block Multi-headed Attention
        """
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim = self.dim_embedding,
            dropout = self.dropout_rate/(4),
            num_heads = 1)

        
        """
            @block Fully-connected NN
        """
        self.fc = nn.Sequential(
            nn.Linear(
                self.dim_hidden,
                self.dim_fc_layer_1
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.dim_fc_layer_1,
                self.dim_fc_layer_2
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features = self.dim_fc_layer_2),
            nn.Linear(
                self.dim_fc_layer_2,
                self.n_classes
            ),
            nn.Softmax(dim = 0)
        )
        
    def forward(self, x, hidden = None):
        vid, movenet_x = x
        vid = vid.to(self.device)
        movenet_x = movenet_x.to(self.device, dtype=torch.float)
        out = None 
        hidden = None 
        
        for t in range(self.n_frames):
            effective_frames = None
            with torch.no_grad():
                frame_features = self.base_model(vid[:, t, :, :, :])
                effective_frames = frame_features

            if self.use_movenet == True:
                appended = []
                for i, feature in enumerate(frame_features):
                    features = torch.cat([frame_features[i], movenet_x[i, t,:]])
                    appended.append(features)
                appended = torch.stack(appended)
                appended = appended.to(self.device)
                effective_frames = appended

            out, hidden = self.lstm_layer(
                effective_frames.unsqueeze(0),
                hidden
            )
            
        
        attn_output, attn_output_weights = self.multihead_attn(out[ : , : , :self.dim_embedding], hidden[0], hidden[1])
        out = attn_output
            
        return self.fc(out[-1, :, :])
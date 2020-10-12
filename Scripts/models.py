import torch
from torch import nn
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    
    """
    Encoder
    """
    
    def __init__(self, encoder_size = 14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoder_size
        resnet = torchvision.models.resnet101(pretrained=True)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoder_size, encoder_size))
        
        self.fine_tune()
    
    
    def forward(self, images):
        out = self.resnet(images)     # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1) # (batch_size, encoded_image_size, encoded_image_size, 2048)
        
        return out
    
    
    def fine_tune(self, fine_tune = True):
        
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        param encoder_dim: feature size of encoded images
        param decoder_dim: size of decoder's RNN
        param attention_dim: size of the attention_dim
        """
        super(Attention, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim) # linear layer to transform encoded_image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim) # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1) # linear layer to calculate values to be softmaxed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1) # softmax layer to calculate weights
        
        
    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propogation
        
        param encoder_out: encoded_images, tensor with shape: (batch_size, num_pixels, encoder_dim)
        param decoder_hidden: previous decoder output, tensor with shape: (batch_size, decoder_dim)
        :returns: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)    # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden) # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) #(batch_size, num_pixels)
        alpha = self.softmax(att) #(batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1)
        
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """
    
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim = 2048, dropout = 0.5):
        """
        param attention_dim: size of attention network
        param embed_dim: embedding size
        param decoder_dim: size of decoder's RNN
        param vocab_size: size of vocabulary
        param encoder_dim: feature size of encoded images
        param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.dropout = dropout
        
        self.Attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim) # embedding layer
        self.dropout = nn.Dropout(self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias = True) #decoding LSTM cell
        self.init_h = nn.Linear(encoder_dim, decoder_dim) # initial hidden state of LSTM
        self.init_c = nn.Linear(encoder_dim, decoder_dim) # initial cell state of LSTM
        self.f_beta = nn.Linear(decoder_dim, encoder_dim) # layer to create sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size) # linear layer to find scores over vocabulary
        
        self.init_weights()
        
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    
    def load_pretrained_weights(self, embeddings):
        """
        Loads layer with pretrained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
        
        
    def fine_tune_embedding(self, fine_tune = True):
        """
        Allow fine tuning of embedding layer? Only makes sense not to allow when using pretrained
        embeddings
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
            
    
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :returns: hidden state, cell state
        """
        
        mean_encoder_out = encoder_out.mean(dim = 1)
        h = self.init_h(mean_encoder_out) #(batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out) #(batch_size, decoder_dim)
        
        return h, c
        
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation
        
        param encoder_out: encoded images, tensor with dimensions (batch_size, enc_image_size, enc_image_size, encoder_dim)
        param encoded_captions: encoded_captions: tensor with dimensions (batch_size, max_caption_length)
        param caption_lengths: length of captions, tensor with shape (batch_size, 1)
        :returns: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(3)
        vocab_size = self.vocab_size
        
        # Flatten image:
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort inputs by decreasing dimension:
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim = 0, descending = True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding: 
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        # encoded_captions: (batch_size, max_caption_length)
        embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)
        
        #Initalize LSTM state
        h, c = self.init_hidden_state(encoder_out) #(batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon 
        # as we generate <end>. So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.Attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


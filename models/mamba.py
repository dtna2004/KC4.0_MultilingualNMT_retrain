import torch
import torch.nn as nn
import torchtext.data as data
import copy, time, io
import numpy as np

from modules.prototypes import Config as DefaultConfig
from modules.loader import DefaultLoader, MultiLoader
from modules.config import MultiplePathConfig as Config
from modules.inference import strategies
from modules import constants as const
from modules.optim import optimizers, ScheduledOptim

import utils.save as saver
from utils.decode_old import create_masks, translate_sentence
from utils.loss import LabelSmoothingLoss
from utils.metric import bleu, bleu_batch_iter, bleu_single, bleu_batch

class MambaBlock(nn.Module):
    """
    Mamba block implementation (State Space Model)
    """
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)
        
        # Projection up to higher dimension
        self.proj_up = nn.Linear(d_model, self.d_inner)
        
        # State Space Parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        # Input projection to get parameters
        self.proj_params = nn.Sequential(
            nn.Linear(d_model, self.d_inner * 2),
            nn.SiLU()
        )
        
        # Output projection
        self.proj_down = nn.Linear(self.d_inner, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len, d_model]
        residual = x
        
        # Layer normalization
        x = self.norm(x)
        
        # Project up
        x_up = self.proj_up(x)  # [batch_size, seq_len, d_inner]
        
        # Compute parameters
        params = self.proj_params(x)  # [batch_size, seq_len, d_inner*2]
        delta, gating = torch.chunk(params, 2, dim=-1)
        delta = torch.sigmoid(delta)  # Convert to (0, 1)
        
        # Prepare state
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        
        outputs = []
        
        # Sequential computation (can be optimized with custom CUDA kernel in real implementation)
        for i in range(seq_len):
            # Current input
            u = x_up[:, i]  # [batch_size, d_inner]
            
            # Update state using discrete state space equation
            h = h * torch.exp(delta[:, i].unsqueeze(-1) * self.A.unsqueeze(0)) + u.unsqueeze(-1) * self.B.unsqueeze(0)
            
            # Compute output
            y = (h * self.C.unsqueeze(0)).sum(dim=-1) + u * self.D
            
            # Apply gating
            y = y * torch.sigmoid(gating[:, i])
            
            outputs.append(y)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [batch_size, seq_len, d_inner]
        
        # Project down
        y = self.proj_down(y)  # [batch_size, seq_len, d_model]
        
        # Apply dropout and residual connection
        y = self.dropout(y)
        y = y + residual
        
        return y

class MambaEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, d_state=16, expand_factor=2, dropout=0.1, max_seq_length=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (simple learnable embeddings for Mamba)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None, **kwargs):
        # src: [batch_size, src_len]
        # src_mask: [batch_size, 1, src_len]
        
        # Get sequence length
        seq_len = src.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(src) * (self.embedding.embedding_dim ** 0.5)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x

class MambaDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, d_state=16, expand_factor=2, dropout=0.1, max_seq_length=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (simple learnable embeddings for Mamba)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Self-attention Mamba blocks
        self.self_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor, dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention layers for encoder-decoder attention
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, trg, memory, src_mask=None, trg_mask=None, output_attention=False, **kwargs):
        # trg: [batch_size, trg_len]
        # memory: [batch_size, src_len, d_model] (output from encoder)
        
        # Get sequence length
        seq_len = trg.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(trg) * (self.embedding.embedding_dim ** 0.5)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        attentions = []
        
        for i in range(len(self.self_layers)):
            # Self-attention with causal mask handled by Mamba
            residual = x
            x = self.self_layers[i](x)
            x = residual + x
            
            # Cross-attention with encoder outputs
            residual = x
            x = self.layer_norms1[i](x)
            x_attn, attn = self.cross_attn_layers[i](
                query=x, 
                key=memory, 
                value=memory, 
                key_padding_mask=~src_mask.squeeze(1) if src_mask is not None else None,
                need_weights=output_attention
            )
            x = residual + x_attn
            
            # Feed-forward
            residual = x
            x = self.ff_layers[i](self.layer_norms2[i](x))
            x = residual + x
            
            if output_attention:
                attentions.append(attn)
        
        # Final layer normalization
        x = self.norm(x)
        
        if output_attention:
            return x, attentions
        return x

class Mamba(nn.Module):
    """
    Implementation of a Mamba-based architecture for neural machine translation.
    Combines State Space Models with cross-attention for encoder-decoder architecture.
    """
    def __init__(self, mode=None, model_dir=None, config=None):
        super().__init__()

        # Use specific config file if provided otherwise use the default config instead
        self.config = DefaultConfig() if(config is None) else Config(config)
        opt = self.config
        self.device = opt.get('device', const.DEFAULT_DEVICE)

        if('train_data_location' in opt or 'train_data_location' in opt.get("data", {})):
            # monolingual data detected
            data_opt = opt if 'train_data_location' in opt else opt["data"]
            self.loader = DefaultLoader(data_opt['train_data_location'], eval_path=data_opt.get('eval_data_location', None), language_tuple=(data_opt["src_lang"], data_opt["trg_lang"]), option=opt)
        elif('data' in opt):
            # multilingual data with multiple corpus in [data][train] namespace
            self.loader = MultiLoader(opt["data"]["train"], valid=opt["data"].get("valid", None), option=opt)
            
        # input fields
        self.SRC, self.TRG = self.loader.build_field(lower=opt.get("lowercase", const.DEFAULT_LOWERCASE))

        # initialize dataset and by proxy the vocabulary
        if(mode == "train"):
            # training flow, necessitate the DataLoader and iterations
            self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_path=model_dir)
        elif(mode == "eval"):
            # evaluation flow, which only require valid_iter
            self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_path=model_dir)
        elif(mode == "infer"):
            # inference, require pickled model and vocab in the path
            self.loader.build_vocab(self.fields, model_path=model_dir)
        else:
            raise ValueError("Unknown model's mode: {}".format(mode))

        # define the model
        src_vocab_size, trg_vocab_size = len(self.SRC.vocab), len(self.TRG.vocab)
        d_model, N, dropout = opt['d_model'], opt['n_layers'], opt['dropout']
        
        # get max lengths
        train_ignore_length = self.config.get("train_max_length", const.DEFAULT_TRAIN_MAX_LENGTH)
        input_max_length = self.config.get("input_max_length", const.DEFAULT_INPUT_MAX_LENGTH)
        infer_max_length = self.config.get('max_length', const.DEFAULT_MAX_LENGTH)
        encoder_max_length = max(input_max_length, train_ignore_length)
        decoder_max_length = max(infer_max_length, train_ignore_length)
        
        # Get Mamba specific params (defaults if not specified)
        d_state = opt.get('d_state', 16)
        expand_factor = opt.get('expand_factor', 2)
        
        # Initialize Mamba encoder and decoder
        self.encoder = MambaEncoder(
            src_vocab_size, d_model, N, 
            d_state=d_state, 
            expand_factor=expand_factor, 
            dropout=dropout, 
            max_seq_length=encoder_max_length
        )
        
        self.decoder = MambaDecoder(
            trg_vocab_size, d_model, N, 
            d_state=d_state, 
            expand_factor=expand_factor, 
            dropout=dropout, 
            max_seq_length=decoder_max_length
        )
        
        self.out = nn.Linear(d_model, trg_vocab_size)

        # load the beamsearch obj with preset values
        decode_strategy_class = strategies[opt.get('decode_strategy', const.DEFAULT_DECODE_STRATEGY)]
        decode_strategy_kwargs = opt.get('decode_strategy_kwargs', const.DEFAULT_STRATEGY_KWARGS)
        self.decode_strategy = decode_strategy_class(self, infer_max_length, self.device, **decode_strategy_kwargs)

        self.to(self.device)

    def load_checkpoint(self, model_dir, checkpoint=None, checkpoint_idx=0):
        """Attempt to load past checkpoint"""
        if(checkpoint is not None):
            saver.load_model(self, checkpoint)
            self._checkpoint_idx = checkpoint_idx
        else:
            if model_dir is not None:
                # load the latest available checkpoint, overriding the checkpoint value
                checkpoint_idx = saver.check_model_in_path(model_dir)
                if(checkpoint_idx > 0):
                    print("Found model with index {:d} already saved.".format(checkpoint_idx))
                    saver.load_model_from_path(self, model_dir, checkpoint_idx=checkpoint_idx)
                else:
                    print("No checkpoint found, start from beginning.")
                    checkpoint_idx = -1
            else:
                print("No model_dir available, start from beginning.")
                # train the model from begin
                checkpoint_idx = -1
            self._checkpoint_idx = checkpoint_idx
            

    def forward(self, src, trg, src_mask, trg_mask, output_attention=False):
        """Run a full model with specified source-target batched set of data"""
        e_outputs = self.encoder(src, src_mask)
        d_output, attn = self.decoder(trg, e_outputs, src_mask, trg_mask, output_attention=True)
        output = self.out(d_output)
        if(output_attention):
            return output, attn
        else:
            return output
 
    def train_step(self, optimizer, batch, criterion):
        """
        Perform one training step.
        """
        self.train()
        opt = self.config
        
        # move data to specific device's memory
        src = batch.src.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))
        trg = batch.trg.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))

        trg_input = trg[:, :-1]
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
        ys = trg[:, 1:].contiguous().view(-1)

        # create mask and perform network forward
        src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, opt.get('device', const.DEFAULT_DEVICE))
        preds = self(src, trg_input, src_mask, trg_mask)
        
        # perform backprogation
        optimizer.zero_grad()
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        loss.backward()
        optimizer.step_and_update_lr()
        loss = loss.item()
        
        return loss    

    def validate(self, valid_iter, criterion, maximum_length=None):
        """Compute loss in validation dataset."""
        self.eval()
        opt = self.config
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
    
        with torch.no_grad():
            total_loss = []
            for batch in valid_iter:
                # load model into specific device (GPU/CPU) memory  
                src = batch.src.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))
                trg = batch.trg.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))
                if(maximum_length is not None):
                    src = src[:, :maximum_length[0]]
                    trg = trg[:, :maximum_length[1]-1] # using partials
                trg_input = trg[:, :-1]
                ys = trg[:, 1:].contiguous().view(-1)

                # create mask and perform network forward
                src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, opt.get('device', const.DEFAULT_DEVICE))
                preds = self(src, trg_input, src_mask, trg_mask)

                # compute loss on current batch
                loss = criterion(preds.view(-1, preds.size(-1)), ys)
                loss = loss.item()
                total_loss.append(loss)
    
        avg_loss = np.mean(total_loss)
        return avg_loss

    def translate_sentence(self, sentence, device=None, k=None, max_len=None, debug=False):
        """
        Receive a sentence string and output the prediction generated from the model.
        NOTE: sentence input is a list of tokens instead of string due to change in loader.
        """
        self.eval()
        if device is None:
            device = self.device
        if k is None:
            k = self.config.get('decode_strategy_kwargs', {}).get('beam_size', 5)
        if max_len is None:
            max_len = self.config.get('max_length', const.DEFAULT_MAX_LENGTH)

        return translate_sentence(sentence, self, self.SRC, self.TRG, device, k, max_len, debug=debug)

    def translate_batch_sentence(self, sentences, src_lang=None, trg_lang=None, output_tokens=False, batch_size=None):
        """
        Receive a list of sentences and translate them. All sentences should be in the same language.
        """
        self.eval()
        if batch_size is None:
            batch_size = self.config.get("eval_batch_size", const.DEFAULT_EVAL_BATCH_SIZE)
            
        return [self.translate_sentence(sent, output_list_of_tokens=output_tokens) for sent in sentences]

    def translate_batch(self, batch_sentences, src_lang=None, trg_lang=None, output_tokens=False, input_max_length=None):
        """
        Translate a batch of sentences using the previously defined batch iterator.
        Does not support multilingual translation yet.
        """
        self.eval()
        self._check_infer_params(src_lang, trg_lang)
        if(input_max_length is None):
            input_max_length = self.config.get("input_max_length", const.DEFAULT_INPUT_MAX_LENGTH)
        return self.translate_batch_sentence(batch_sentences, src_lang=src_lang, trg_lang=trg_lang, output_tokens=output_tokens)

    def run_train(self, model_dir=None, config=None):
        """
        Kịch bản huấn luyện, train mô hình theo đã nạp iterator.
        """
        if(config is not None):
            self.config = Config(config)
        opt = self.config
        
        # Load the model and corresponding vocab
        train_iter, valid_iter = self.train_iter, self.valid_iter

        # Create the criterion, optimizer and the actual model
        criterion = LabelSmoothingLoss(classes=len(self.TRG.vocab), padding_idx=self.TRG.vocab.stoi['<pad>'], smoothing=opt.get('label_smoothing', 0.0))
        
        # Create the corresponding optimizer
        optimizer_class = optimizers[opt.get('optimizer', "Adam")]
        optimizer = optimizer_class(self.parameters(), lr=opt.get("lr", 0.001), **opt.get("optimizer_params", {}))

        # Scheduler for LR policy
        optimizer = ScheduledOptim(
            optimizer=optimizer,
            init_lr=opt.get("lr", 0.001),
            d_model=opt.get('d_model', 512),
            n_warmup_steps=opt.get('n_warmup_steps', 4000)
        )

        # Set all prerequisite for training
        printevery = opt.get('printevery', 100)
        total_loss = 0
        start = time.time()

        # Set up some variables
        self._checkpoint_idx = saver.check_model_in_path(model_dir)
        model = self.to(opt.get('device', const.DEFAULT_DEVICE))
        
        # train loop
        for epoch in range(self._checkpoint_idx+1, opt.get('epochs', 30)):
            print("Epoch {:d}".format(epoch))
            # Get iterator and corresponding losses
            for i, batch in enumerate(train_iter):
                loss = self.train_step(optimizer, batch, criterion)
                total_loss += loss

                if (i + 1) % printevery == 0:
                    avg_loss = float(total_loss/printevery)
                    elapsed = time.time() - start
                    print("Epoch: {:d}, Step: {:d}, Avg Loss: {:.4f}, Elapsed: {:.2f}s".format(epoch, i+1, avg_loss, elapsed))
                    total_loss = 0
                    start = time.time()

            valid_loss = self.validate(valid_iter, criterion, 
                maximum_length=(
                    opt.get("input_max_length", const.DEFAULT_INPUT_MAX_LENGTH),
                    opt.get("max_length", const.DEFAULT_MAX_LENGTH)
                )
            )
            
            # Compute BLEU
            #bleu_score = bleu_batch_iter(self, valid_iter, self.SRC, self.TRG, opt.get('device', const.DEFAULT_DEVICE))

            print("VALIDATION - Valid Loss: {:.4f}".format(valid_loss))
            
            # We must save the model at this point
            if epoch % opt.get('save_checkpoint_epochs', const.DEFAULT_SAVE_CHECKPOINT_EPOCH) == 0:
                saver.save_model(self, model_dir, epoch, valid_loss, train_loss=avg_loss)

    def run_eval(self, model_dir=None, config=None):
        """
        Evaluation runs, compute the BLEU score of the model by comparing with TRG tokens
        """
        if(config is not None):
            self.config = Config(config)
        opt = self.config
        
        # We don't need to do anything as validation is run during training. Just have a different model signature here
        valid_iter = self.valid_iter
        bleu_score = bleu_batch_iter(self, valid_iter, self.SRC, self.TRG, self.device)
        print("BLEU Score: {:.4f}".format(bleu_score))
        return bleu_score

    def run_infer(self, features_file, predictions_file, src_lang=None, trg_lang=None, config=None, batch_size=None):
        """
        Inference run, use the model to translate source in another file
        """
        if(config is not None):
            self.config = Config(config)

        opt = self.config
        
        # default batch size if not defined
        if batch_size is None:
            batch_size = opt.get("eval_batch_size", const.DEFAULT_EVAL_BATCH_SIZE)

        self._check_infer_params(src_lang, trg_lang)
        
        source_sentences = open(features_file, encoding='utf-8').readlines()
        # preprocessed source sentences
        src_sentences = [self.SRC.preprocess(sent) for sent in source_sentences]
        translations = []

        # create minibatch to avoid possible OOM
        minibatched_src = [src_sentences[i:i+batch_size] for i in range(0, len(src_sentences), batch_size)]

        for batch in minibatched_src:
            print(".", end='', flush=True)
            translations.extend(self.translate_batch(batch, src_lang=src_lang, trg_lang=trg_lang))
        print()  # create a linebreak
        
        with open(predictions_file, 'w', encoding='utf-8') as out:
            for sent in translations:
                out.write(sent + '\n')

    # interface for interacting with the model, useful for loading the module elsewhere
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def to_logits(self, inputs): # function to include the logits. TODO use this in inference fns as well
        return self.out(inputs)

    def prepare_serve(self, serve_path, model_dir=None, check_trace=True, **kwargs):
        """Compile the model for serving"""
        if model_dir is not None:
            # model_dir also contains the hyperparams and vocabs so we need to load it
            print("Loading checkpoint for serving...")
            checkpoint_idx = saver.check_model_in_path(model_dir)
            if(checkpoint_idx <= 0):
                raise ValueError("Model at path {:s} not saved yet.".format(model_dir))
            saver.load_model_from_path(self, model_dir, checkpoint_idx=checkpoint_idx)

        self.eval()
        # Prepare 2 junk input for tracing purpose
        src = torch.LongTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        trg = torch.LongTensor([[0, 1, 2], [0, 1, 2]])
        src_mask = create_masks(src, None, 0, 0, device=self.device)[0]
        trg_mask = create_masks(None, trg, 0, 0, device=self.device)[1]

        # Create a script module out of self
        print("Tracing model with junk input...")
        traced_model = torch.jit.trace(self, (src, trg, src_mask, trg_mask))

        if(check_trace):
            print("Checking that traced model match model for the junk input...")
            check1 = self(src, trg, src_mask, trg_mask)
            check2 = traced_model(src, trg, src_mask, trg_mask)
            assert torch.allclose(check1, check2)
    
        print("Saving model to {:s}...".format(serve_path))
        traced_model.save(serve_path)
    
    def _check_infer_params(self, src_lang, trg_lang):
        """Check the parameters of inference"""
        pass

    @property
    def fields(self):
        return (self.SRC, self.TRG) 
import torch
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight and bias parameters without initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        return torch.nn.functional.linear(input, self.weight, self.bias)


class MLP(torch.nn.Module):

    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        # Check if the input size is greater than 0
        assert input_size > 0, "The current size is not greater than 0"

        layers = []
        current_size = input_size
        
        # Build hidden layers with our Linear class
        # Hidden sizes is not empty (checked in the constructor)
        for hidden_size in hidden_sizes:
            # Check if the current size is not 0
            assert hidden_size > 0, "The hidden size is not greater than 0"
            layers.append(Linear(current_size, hidden_size))
            current_size = hidden_size
        
        # Create ModuleList for hidden layers
        hidden_layers = nn.ModuleList(layers)
        # Create output layer with our Linear class
        output_layer = Linear(current_size, num_classes)
        
        return hidden_layers, output_layer
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        if activation == 'relu':
            return torch.relu(inputs)
        elif activation == 'tanh':
            return torch.tanh(inputs)
        elif activation == 'sigmoid':
            return torch.sigmoid(inputs)
        else:
            raise ValueError(f"Invalid activation function: {activation}")
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        # Glorot normal initialization for weights
        nn.init.xavier_normal_(module.weight)
        # Initialize bias to zero
        nn.init.zeros_(module.bias)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        # Flatten the images
        batch_size = images.size(0)
        x = images.view(batch_size, -1)
        
        # Pass through hidden layers with activation
        for layer in self.hidden_layers:
            x = self.activation_fn(self.activation, layer(x))
        
        # Pass through output layer
        logits = self.output_layer(x)
        
        return logits

    def get_gradient_flow(self) -> dict:
        gradients = {
            'hidden_layers': [],
            'output_layer': None
        }
        
        for idx, layer in enumerate(self.hidden_layers):
            if layer.weight.grad is not None:
                grad_mean = layer.weight.grad.abs().mean().item()
                gradients['hidden_layers'].append(grad_mean)
            
        if self.output_layer.weight.grad is not None:
            grad_mean = self.output_layer.weight.grad.abs().mean().item()
            gradients['output_layer'] = grad_mean
            
        return gradients

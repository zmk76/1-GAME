from torch import nn
import torch
import torch.nn.functional as F

class MyLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        # print("init weight:")
        # print(self.weight)
        # asdfasdfadf

        # nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)




class MyNormalLinear(nn.Linear):
    # pass
    def reset_parameters(self) -> None:
        # nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MyPReLU(nn.Module):

    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()

        # use alpha instead of weight
        self.alpha = nn.parameter.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)
        
def create_act(name=None):
    if name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


# class MyLinear(nn.Linear):
#     def reset_parameters(self) -> None:
#         nn.init.xavier_normal_(self.weight)
#         nn.init.zeros_(self.bias)


def get_activation(activation):
    if activation == "prelu":
        return MyPReLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation is None or activation == "none":
        return torch.nn.Identity()
    else:
        raise NotImplementedError()
    



class MyMLP(nn.Module):
    def __init__(self, in_channels, units_list, activation, drop_rate, bn, output_activation, output_drop_rate, output_bn, ln=False, output_ln=False):
        super().__init__()

        layers = []
        units_list = [in_channels] + units_list  # Add in_channels to the list of units

        for i in range(len(units_list) - 1):
            layers.append(MyLinear(units_list[i], units_list[i+1]))  # Create a linear layer
            # layers.append(MyNormalLinear(units_list[i], units_list[i+1]))  # Create a linear layer


            if i < len(units_list) - 2:
                if bn:
                    layers.append(nn.BatchNorm1d(units_list[i+1]))  # Add a batch normalization layer

                if ln:
                    layers.append(nn.LayerNorm(units_list[i+1]))
                
                layers.append(get_activation(activation))  # Add the PReLU activation function
                layers.append(nn.Dropout(drop_rate))
            else:
                if output_bn:
                    layers.append(nn.BatchNorm1d(units_list[i+1]))

                if output_ln:
                    layers.append(nn.LayerNorm(units_list[i+1]))

                layers.append(get_activation(output_activation))  # Add the PReLU activation function
                layers.append(nn.Dropout(output_drop_rate))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

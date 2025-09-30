import torch
from torch import nn
import torch.autograd.profiler as profiler

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, init_fct=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


# Aggregation Block
class AggregationBlock(nn.Module):
    def __init__(self, sizes_in, pre_dims, agg_fct="sum", agg_dim=1, beta=0.0, init_fct=None):
        """
        Aggregate and preprocess several inputs for next layers
        :param sizes_in (dict): input_name -> input_dim
        :param pre_dims (dict): input_name -> linear preprocessing dims (list)
        :param agg_fct (str): aggregation function used
        :param agg_dim (int): dimension to aggregate along
        """

        super().__init__()
        assert all(k in sizes_in for k in pre_dims)
        # All preprocessing done on known inputs
        self.sizes_in = sizes_in
        self.pre_dims = pre_dims
        self.agg_fct = agg_fct

        def create_lin(dims, init_fct):
            lin_layers = [nn.Linear(dims[0], dims[1])]
            for i in range(1, len(dims) - 1):
                lin_layers.append(nn.ReLU(True))
                lin_layers.append(nn.Linear(dims[i], dims[i + 1]))

                if init_fct is None:
                    # init
                    nn.init.constant_(lin_layers[-1].bias, 0.0)
                    nn.init.kaiming_normal_(lin_layers[-1].weight, a=0, mode="fan_in")
                elif init_fct == "zero":
                    nn.init.constant_(lin_layers[-1].bias, 0.0)
                    nn.init.zeros_(lin_layers[-1].weight)

            """
            for i in range(len(lin_layers)):
                nn.init.constant_(lin_layers[i].bias, 0.0)
                nn.init.kaiming_normal_(lin_layers[i].weight, a=0, mode="fan_in")
                """

            return nn.Sequential(*lin_layers)

        common_last_dim = None
        for input_name, input_dim in sizes_in.items():
            last_dim = input_dim
            if input_name in pre_dims:
                setattr(self, f"pre_lin_{input_name}", create_lin(pre_dims[input_name], init_fct))
                last_dim = pre_dims[input_name][-1]
            if common_last_dim is None:
                common_last_dim = last_dim
            if agg_fct in ["sum", "mean"] and common_last_dim != last_dim:
                raise IOError("Incompatible aggregation dims")
        if "uv_src_feats" in sizes_in:
            setattr(
                self,
                f"pre_lin_uv_src_feats_comb",
                create_lin((2 * pre_dims["uv_src_feats"][1], pre_dims["uv_src_feats"][1]), init_fct),
            )
        self.agg_dim = agg_dim

    def forward(self, inputs):
        with profiler.record_function("aggblock"):
            # preprocess inputs for aggregation
            agg_data = dict()
            for input_name, input_data in inputs.items():
                if input_name in self.pre_dims:
                    agg_data[input_name] = getattr(self, f"pre_lin_{input_name}")(input_data)
                else:
                    agg_data[input_name] = input_data
            # aggregate data
            if "uv_src_feats" in agg_data:
                # HARDCODED SPECIAL TREATMENT FOR shared features
                agg_feats_1 = torch.mean(
                    torch.stack([v for k, v in agg_data.items() if k != "uv_src_feats"], self.agg_dim), self.agg_dim
                )
                agg_features = torch.cat([agg_feats_1, agg_data["uv_src_feats"]], 1)
                agg_features = getattr(self, f"pre_lin_uv_src_feats_comb")(agg_features)

            if self.agg_fct == "sum":
                agg_features = torch.sum(torch.stack(list(agg_data.values()), self.agg_dim), self.agg_dim)
            elif self.agg_fct == "mean":
                agg_features = torch.mean(torch.stack(list(agg_data.values()), self.agg_dim), self.agg_dim)
            elif self.agg_fct == "cat":
                agg_features = torch.cat(list(agg_data.values()), self.agg_dim)
        return agg_features


class BottleneckBlock(nn.Module):
    def __init__(self, in_dim, dim, activation=None):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.activation = activation
        self.in_layer = nn.Linear(self.in_dim, dim)
        self.out_layer = nn.Linear(dim, self.in_dim)

    def forward(self, x):
        x_bottleneck = self.in_layer(x)
        if self.activation is not None:
            if self.activation == "relu":
                x_bottleneck = nn.ReLU(True)(x_bottleneck)
            else:
                raise NotImplementedError
        return self.out_layer(x_bottleneck), x_bottleneck


class MlpResNet(nn.Module):
    """
    Represents a MLP;
    Original code from IGR
    """

    def __init__(
        self,
        d_in,
        dims,
        d_out,
        injections=None,
        agg_fct="sum",
        bottleneck=None,
        init_fct=None,
        add_out_lvl=None,
        add_out_dim=None,        
        input=None,
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param geometric_init if true, uses geometric initialization
               (to SDF of sphere)
        :param radius_init if geometric_init, then SDF sphere will have
               this radius
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        :param output_init_gain output layer normal std, only used for
                                output dimension >= 1, when d_out >= 1
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()

        self.num_layers = len(dims)
        self.dims = dims
        self.injections = injections if injections is not None else {}  # {layer_level: injection_dimensions}
        self.bottleneck = bottleneck  # (layer, dim, activation)
        self.input = input

        # building blocks
        for layer in range(self.num_layers):
            if layer == 0:
                sizes_in = {"x": d_in}
                pre_dims = {"x": [d_in, dims[0]]}
            else:
                sizes_in = {"x": dims[layer]}
                pre_dims = {}
            for input_name, inj_dims in self.injections.get(layer, {}).items():
                if isinstance(inj_dims, int):
                    sizes_in[input_name] = inj_dims
                else:
                    sizes_in[input_name] = inj_dims[0]
                    pre_dims[input_name] = inj_dims

            agg_block = AggregationBlock(sizes_in, pre_dims, agg_fct, agg_dim=1, init_fct=init_fct)
            lin_block = ResnetBlockFC(dims[layer], init_fct=init_fct)

            setattr(self, f"agg_{layer}", agg_block)
            setattr(self, f"lin_{layer}", lin_block)
            if self.bottleneck is not None and self.bottleneck[0] == layer:
                # insert bottleneck layer
                bottleneck_block = BottleneckBlock(
                    dims[layer], self.bottleneck[1], self.bottleneck[2] if len(self.bottleneck) > 2 else None,
                )
                setattr(self, f"lin_bottleneck", bottleneck_block)

        self.lin_out = nn.Linear(dims[-1], d_out)
        if init_fct is not None:
            if init_fct == "zero":
                nn.init.constant_(self.lin_out.bias, 0.0)
                nn.init.zeros_(self.lin_out.weight)

        self.add_out_lvl = add_out_lvl
        self.add_out_dim = add_out_dim
        if add_out_dim is not None:
            self.lin_add_out = nn.Linear(dims[self.add_out_lvl], add_out_dim)

    def forward(self, x, inj_data=None, add_out_lvl=None):
        if add_out_lvl is None:
            add_out_lvl = self.add_out_lvl

        add_output = None
        for layer in range(0, self.num_layers):
            agg_layer = getattr(self, f"agg_{layer}")
            lin_layer = getattr(self, f"lin_{layer}")
            # input according to self.injections
            agg_data = {"x": x}
            for input_name in self.injections.get(layer, []):
                agg_data[input_name] = inj_data[input_name]
            x = agg_layer(agg_data)
            x = lin_layer(x)
            if layer == add_out_lvl:
                add_output = x
            if self.bottleneck is not None and self.bottleneck[0] == layer:
                x, x_bottleneck = getattr(self, f"lin_bottleneck")(x)

        x = self.lin_out(x)
        if add_out_lvl is None:
            if self.bottleneck is not None:

                return x, x_bottleneck
            else:
                return x
        if self.add_out_dim is not None:
            add_output = self.lin_add_out(add_output)
        if self.bottleneck is not None:
            return x, add_output, x_bottleneck
        return x, add_output


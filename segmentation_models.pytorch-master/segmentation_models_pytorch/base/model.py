import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder_1)
        init.initialize_head(self.segmentation_head_1)
        init.initialize_decoder(self.decoder_1)
        init.initialize_head(self.segmentation_head_1)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output_1 = self.decoder_1(*features)

        decoder_output_2 = self.decoder_1(*features)


        masks_1 = self.segmentation_head_1(decoder_output_1)
        masks_2 = self.segmentation_head_2(decoder_output_2)


        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks_1,masks_2, labels

        return masks_1

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

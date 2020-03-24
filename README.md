# image-to-image Translation using MXNet and Generative Networks 

Implementation of GAN models for Image to Image translation using MXNet:

* Deep Convolution GAN with image auto-encoder.
* [pixel2pixel.py](src/library/pixel2pixel.py): Pixel-to-Pixel GAN.

### Deep Convolution GAN with VGG16 source image encoder

To run DCGan using 
the facade dataset, run the following command:

```bash
python demo/dcgan_train.py
```

The demo/dcgan_train.py sample codes are shown below:

```
def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)
```
```
def main():
    sys.path.append(patch_path('..'))

    output_dir_path = patch_path('models')

    logging.basicConfig(level=logging.DEBUG)

    from src.library.dcgan import DCGan
    from src.data.facades_data_set import load_image_pairs

    ctx = mx.cpu()
    img_pairs = load_image_pairs(patch_path('data/facades'))
    gan = DCGan(model_ctx=ctx)
    gan.random_input_size = 24

    gan.fit(image_pairs=img_pairs, model_dir_path=output_dir_path)


if __name__ == '__main__':
    main()
```


The trained models will be saved into demo/models folder with prefix "dcgan-*"

To run the trained models to generate new images:

```bash
python demo/dcgan_generate.py
```

The demo/dcgan_generate.py sample codes are shown below:

```python
def main():
    sys.path.append(patch_path('..'))
    output_dir_path = patch_path('output')
    model_dir_path = patch_path('models')

    from src.library.dcgan import DCGan
    from src.data.facades_data_set import load_image_pairs
    from src.library.image_utils import load_image, visualize, save_image

    img_pairs = load_image_pairs(patch_path('data/facades'))

    ctx = mx.cpu()
    gan = DCGan(model_ctx=ctx)
    gan.load_model(model_dir_path)

    shuffle(img_pairs)

    for i, (source_img_path, _) in enumerate(img_pairs[:20]):
        source_img = load_image(source_img_path, 64, 64)
        target_img = gan.generate(source_image_path=source_img_path, filename=str(i)+'.png', output_dir_path=output_dir_path)
        img = mx.nd.concat(source_img.as_in_context(gan.model_ctx), target_img, dim=2)
        visualize(img)
        img = ((img.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        save_image(img, os.path.join(output_dir_path, DCGan.model_name + '-generated-' + str(i) + '.png'))


if __name__ == '__main__':
    main()

```

The Pixel2PixelGan outperforms the DCGan in terms of image translation quality.

Below is some output images generated:

|  ![](images/pixel-2-pixel-gan-generated-1.png) | ![](images/pixel-2-pixel-gan-generated-2.png) |


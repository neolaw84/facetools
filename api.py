from torch import optim
from tqdm.auto import tqdm

import cv2

from .helper import *
from .model.generator import SkipEncoderDecoder, input_noise


def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps, tqdm_length=100, output_path=None):
    DTYPE = torch.FloatTensor
    has_set_device = False
    if torch.cuda.is_available():
        device = 'cuda'
        has_set_device = True
        print("Setting Device to CUDA...")
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
            has_set_device = True
            print("Setting Device to MPS...")
    except Exception as e:
        print(f"Your version of pytorch might be too old, which does not support MPS. Error: \n{e}")
        pass
    if not has_set_device:
        device = 'cpu'
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down = [128] * 5,
        num_channels_up = [128] * 5,
        num_channels_skip = [128] * 5
    ).type(DTYPE).to(device)

    objective = torch.nn.MSELoss().type(DTYPE).to(device)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    print('\nStarting training...\n')

    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_saved

        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        if show_step != 0 and step % show_step == 0:
            output_image = torch_to_np_array(output)
            visualize_sample(image_np, output_image, nrow = 2, size_factor = 10)

        progress_bar.set_postfix(Loss = loss.item())

        optimizer.step()

    output_image = torch_to_np_array(output)
    if show_step != 0:
        visualize_sample(output_image, nrow = 1, size_factor = 10)

    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    if output_path is None: output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')

    pil_image.save(output_path)


def remove_watermark_np(image_np_full, mask_np_full, max_dim=1920, reg_noise=0.03, input_depth=32, lr=0.01, show_step=0, training_steps=300):
    DTYPE = torch.FloatTensor
    def _set_device_if_not():
        if not remove_watermark_np._has_set_device:
            has_set_device = False
            if torch.cuda.is_available():
                device = 'cuda'
                has_set_device = True
                print("Setting Device to CUDA...")
            try:
                if torch.backends.mps.is_available():
                    device = 'mps'
                    has_set_device = True
                    print("Setting Device to MPS...")
            except Exception as e:
                print(f"Your version of pytorch might be too old, which does not support MPS. Error: \n{e}")
                pass
            if not has_set_device:
                device = 'cpu'
                print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
                print('It is recommended to use GPU if possible...')
            remove_watermark_np._has_set_device = True
            remove_watermark_np._device = device
        return remove_watermark_np._device

    device = _set_device_if_not()

    image_np_full = image_np_full.astype(np.float32) / 255.

    mask_np_full = mask_np_full.astype(np.float32) / 255.

    image_np_full = image_np_full.transpose(2, 0, 1)
    
    mask_np_full = mask_np_full.transpose(2, 0, 1)

    if show_step != 0: 
        print('Visualizing mask overlap...')
        visualize_sample(image_np_full, mask_np_full, image_np_full * mask_np_full, nrow = 3, size_factor = 10)

    # Get image dimensions
    height, width = image_np_full.shape[1:]

    # Initialize the output image
    output_img = np.copy(image_np_full)

    # Define patch size
    patch_size = 256

    # Iterate over image patches
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            mask_np = mask_np_full[:, y:y+patch_size, x:x+patch_size]
            if np.all(mask_np > 0.8):
                continue
            image_np = image_np_full[:, y:y+patch_size, x:x+patch_size]
            
            # print('Building the model...')
            generator = SkipEncoderDecoder(
                input_depth,
                num_channels_down = [128] * 5,
                num_channels_up = [128] * 5,
                num_channels_skip = [128] * 5
            ).type(DTYPE).to(device)

            objective = torch.nn.MSELoss().type(DTYPE).to(device)
            optimizer = optim.Adam(generator.parameters(), lr)

            image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
            mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)

            generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

            generator_input_saved = generator_input.detach().clone()
            noise = generator_input.detach().clone()

            # print('\nStarting training...\n')

            # progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

            # for step in progress_bar:
            for step in range(training_steps):
                optimizer.zero_grad()
                generator_input = generator_input_saved

                if reg_noise > 0:
                    generator_input = generator_input_saved + (noise.normal_() * reg_noise)

                output = generator(generator_input)

                loss = objective(output * mask_var, image_var * mask_var)
                loss.backward()

                if show_step != 0 and step % show_step == 0:
                    output_image = torch_to_np_array(output)
                    visualize_sample(image_np, output_image, nrow = 2, size_factor = 5)

                # progress_bar.set_postfix(Loss = loss.item())

                optimizer.step()

            output_image = torch_to_np_array(output)
        
            if show_step != 0:
                visualize_sample(output_image, nrow = 1, size_factor = 5)

            # we take original for the masked value is white 
            # mask_np_full is inverted in preprocess function in helper.py
            output_image[mask_np > 0.8] = image_np[mask_np > 0.8]

            output_img[:, y:y+patch_size, x:x+patch_size] = output_image
            
    return cv2.cvtColor((output_img.transpose(1, 2, 0) * 255.0).astype('uint8'), cv2.COLOR_BGR2RGB)

remove_watermark_np._has_set_device = False 
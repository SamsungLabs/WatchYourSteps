import torch, torchvision
from wys_2d import LocalEditor as WysLocalEditor

#-------------------------------------------------------#
# Path to target image
input_image = "eg-statue.jpg"  
# Text used to guide image translation
edit_text = "Cover the statue in green leaves and vines"
# Parameters
wys_tau      = 0.6 # Mask threshold hyper-parameter
img_size     = 512 # Input image size
s_img, s_txt = 0.9, 8.0 # Guidance strengths for IP2P
#-------------------------------------------------------#

# Build WYS image editor (2D)
wys_editor_model = WysLocalEditor()

# Read and preprocess image
image_tensor0 = torchvision.io.read_image(input_image).float() / 255 * 2.0 - 1
image_tensor  = torchvision.transforms.Resize(size = img_size)(image_tensor0)
print('Loading and processing', input_image, f'({image_tensor0.shape} -> {image_tensor.shape})')

print('Editing image (WYS)')
edited_image, heatmap = wys_editor_model(image_tensor, 
                                         edit = edit_text, 
                                         check_size = False,
                                         scale_txt = s_txt,
                                         scale_img = s_img,
                                         mask_threshold = wys_tau,
                                         return_heatmap = True)

print('Editing image (IP2P)')
edited_image_ip2p, _ = wys_editor_model(image_tensor, 
                                        edit = edit_text, 
                                        check_size = False,
                                        scale_txt = s_txt,
                                        scale_img = s_img,
                                        mask_threshold = 0.0,
                                        return_heatmap = True)                                         

print(f'Saving edit ({edit_text})')
torchvision.utils.save_image(tensor = torch.cat( ( image_tensor.unsqueeze(0).cpu(), 
                                                   edited_image.unsqueeze(0).cpu(), 
                                                   heatmap.expand(-1,3,-1,-1).cpu(),
                                                   edited_image_ip2p.unsqueeze(0).cpu()),
                                                dim = 0), 
                             normalize = True,
                             scale_each = True,
                             fp = 'eg-edited.png')


#
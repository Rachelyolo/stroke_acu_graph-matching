from nilearn import datasets
import nibabel as nib


from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import  matplotlib.pyplot as plt
# example_filename = "yangbin.nii"
#
# img = nib.load(example_filename)
# print(img.dataobj.shape)
# from nilearn import datasets
# destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
# print(destrieux_atlas)
# print(img)
# input()
# atlas_filename = img.maps
# from nilearn import plotting
#
# plotting.plot_roi(atlas_filename, title="Harvard Oxford atlas")
# plotting.show()
# print (img)
# print (img.header['db_name'])
#
# width, height, queue = img.dataobj.shape
#
# OrthoSlicer3D(img.dataobj).show()
#
# num = 1
# for i in range(0, queue, 10):
#     img_arr = img.dataobj[:, :, i]
#     plt.subplot(5, 4, num)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1
#
# plt.show()
from nilearn import datasets

dataset = datasets.fetch_development_fmri(n_subjects=10)

# print basic information on the dataset
print(f"First subject functional nifti image (4D) is at: {dataset.func[0]}")

dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
    "Posterior Cingulate Cortex",
    "Left Temporoparietal junction",
    "Right Temporoparietal junction",
    "Medial prefrontal cortex",
]

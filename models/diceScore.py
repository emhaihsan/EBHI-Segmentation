import torch
from torch import Tensor

# Fungsi untuk menghitung koefisien DICE antara dua tensor input dan target.


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Memastikan ukuran input dan target sama.
    assert input.size() == target.size()
    # Memastikan input adalah tensor 3D atau tidak mengurangi dimensi batch jika tidak diizinkan.
    assert input.dim() == 3 or not reduce_batch_first

    # Menentukan dimensi yang akan dihitung untuk DICE.
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # Menghitung intersection (2 * jumlah elemen positif yang cocok).
    inter = 2 * (input * target).sum(dim=sum_dim)

    # Menghitung jumlah elemen positif pada input dan target.
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # Menghindari pembagian oleh nol dengan menggantikan nol dalam sets_sum dengan inter.
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Menghitung koefisien DICE dengan menambahkan epsilon untuk menghindari pembagian oleh nol.
    dice = (inter + epsilon) / (sets_sum + epsilon)

    # Mengembalikan rata-rata koefisien DICE.
    return dice.mean()

# Fungsi untuk menghitung koefisien DICE multikelas antara dua tensor input dan target.


def multiclass_dice_coef(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Menggunakan fungsi dice_coeff setelah meratakan (flatten) tensor input dan target.
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

# Fungsi untuk menghitung loss (kehilangan) berdasarkan koefisien DICE, dengan opsi multikelas.


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Memilih fungsi yang sesuai berdasarkan apakah multikelas diaktifkan.
    fn = multiclass_dice_coef if multiclass else dice_coeff
    # Loss DICE adalah 1 dikurangi nilai koefisien DICE yang dihitung.
    return 1 - fn(input, target, reduce_batch_first=True)

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class FlourFolderDataset(Dataset):
    """
    Legge HSI in .h5 organizzati per classi: root/class_name/*.h5

    Restituisce la **radianza spettrale** L(λ,x,y) = E(λ) * R(λ,x,y),
    dove R è il cubo letto dal file (assunto in riflettanza) ed E è l'illuminante.

    Output: (x, y) con
        x: torch.Tensor di shape (L, H, W)  [canali spettrali davanti]
        y: indice di classe (int)
        (+ path se return_path=True)
    """

    # Wavelengths di default se assenti nel file: 400..1000 step 5 → 121 canali
    DEFAULT_WAVELENGTHS = np.arange(400.0, 1000.0 + 1e-9, 5.0)

    def __init__(
        self,
        root: str,
        dataset_keys=("patch",),            # chiavi candidate nel file .h5 per il cubo
        exclude_prefixes=("._",),
        dtype=torch.float32,
        hsi_channels_first: bool = False,   # True se HSI salvato come (L,H,W); altrimenti (H,W,L)
        illuminant_mode: str = "planck",    # "planck" | "none" | "array"
        illuminant_T: float = 2856.0,       # K, usato se illuminant_mode="planck"
        illuminant_array: np.ndarray = None,# shape (L,) se illuminant_mode="array"
        normalize_illuminant: bool = True,  # normalizza E(λ) (radianza relativa)
    ):
        super().__init__()
        self.root = root
        self.dataset_keys = dataset_keys
        self.exclude_prefixes = exclude_prefixes
        self.dtype = dtype
        self.hsi_channels_first = hsi_channels_first
        self.illuminant_mode = illuminant_mode
        self.illuminant_T = illuminant_T
        self.illuminant_array = illuminant_array
        self.normalize_illuminant = normalize_illuminant

        # ---- indicizza classi e file ----
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")])
        if not self.classes:
            raise RuntimeError(f"Nessuna classe trovata in {root}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        samples = []
        for cls in self.classes:
            cdir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cdir)):
                if not fname.endswith(".h5"):
                    continue
                if any(fname.startswith(p) for p in self.exclude_prefixes):
                    continue
                samples.append((os.path.join(cdir, fname), self.class_to_idx[cls]))
        if not samples:
            raise RuntimeError(f"Nessun .h5 trovato sotto {root}")
        self.samples = samples

    # ---------- API Dataset ----------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]

        # carica riflettanza e lunghezze d'onda
        cube_R, wl = self._load_h5(path)  # cube_R: (H,W,L), wl: (L,)

        # wavelengths di default se mancano
        if wl is None or wl.size == 0:
            wl = self.DEFAULT_WAVELENGTHS

        # costruisci illuminante E(λ)
        E = self._build_illuminant(wl)  # shape (L,)

        # radianza L(λ,x,y) = E(λ) * R(λ,x,y)
        Lcube = cube_R * E.reshape(1, 1, -1)  # broadcasting su (H,W,L)

        # porta a formato (C,H,W) per PyTorch
        x = torch.from_numpy(np.moveaxis(Lcube, -1, 0)).to(self.dtype)  # radianza
        y = torch.from_numpy(np.moveaxis(cube_R, -1, 0)).to(self.dtype)  # riflettanza

        return x, y

    # ---------- Helpers ----------
    @staticmethod
    def planck_spd(wl_nm: np.ndarray, T: float = 2856.0, normalize: bool = True) -> np.ndarray:
        """Spettro planckiano (Illuminant A ≈ alogena). wl_nm: nm, ritorna E(λ) di shape (L,)."""
        wl_m = wl_nm.astype(np.float64) * 1e-9
        h = 6.62607015e-34
        c = 2.99792458e8
        k = 1.380649e-23
        E = (1.0 / wl_m**5) / (np.exp((h * c) / (wl_m * k * T)) - 1.0)
        if normalize:
            E /= (E.max() + 1e-12)
        return E

    @staticmethod
    def _read_first_present(f, keys):
        for k in keys:
            if k in f and isinstance(f[k], h5py.Dataset):
                return f[k][()]
        return None

    def _load_h5(self, path):
        """
        Restituisce:
          cube_R: riflettanza come (H,W,L)  (float64)
          wl    : (L,) in nm, se presente; altrimenti None
        """
        with h5py.File(path, "r") as f:
            arr = None
            # 1) prova i dataset key in ordine (come prima)
            for k in self.dataset_keys:
                if k in f and isinstance(f[k], h5py.Dataset):
                    arr = f[k][()]
                    break
            # 2) fallback: primo dataset foglia
            if arr is None:
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        arr = f[k][()]
                        break
            if arr is None:
                raise KeyError(f"Nessun dataset valido in {path}")

            # wavelengths se esistono
            wl = None
            for k in ("wavelength", "wavelengths", "lambda", "bands"):
                if k in f and isinstance(f[k], h5py.Dataset):
                    wl = np.asarray(f[k][()], dtype=np.float64).reshape(-1)
                    break

        if arr.ndim != 3:
            raise ValueError(f"atteso 3D, trovato {arr.shape} in {path}")

        # Porta a (H,W,L)
        if self.hsi_channels_first:
            # file salvato come (L,H,W) → (H,W,L)
            if arr.shape[0] != 0:
                cube = np.moveaxis(arr, 0, -1)
            else:
                cube = arr
        else:
            # già (H,W,L) oppure (H,W,C)
            cube = arr

        # wl di default se mancano e se è davvero spettrale (>=3 bande)
        if wl is None and cube.shape[-1] >= 1:
            # se riconosci 121 bande tipiche, usa 400..1000 step 5
            if cube.shape[-1] == 121:
                wl = np.arange(400.0, 1000.0 + 1e-9, 5.0, dtype=np.float64)
            else:
                # placeholder: indici 0..L-1 (non fisici). Meglio fornire wl nei file.
                wl = np.arange(cube.shape[-1], dtype=np.float64)

        return cube.astype(np.float64), wl

    def _build_illuminant(self, wl_target: np.ndarray) -> np.ndarray:
        """
        Costruisce E(λ) su wl_target (nm).
        """
        wl_target = np.asarray(wl_target, dtype=np.float64).reshape(-1)
        L = wl_target.size

        if self.illuminant_mode == "none":
            E = np.ones(L, dtype=np.float64)
        elif self.illuminant_mode == "planck":
            E = self.planck_spd(wl_target, T=self.illuminant_T,
                                normalize=self.normalize_illuminant)
        elif self.illuminant_mode == "array":
            if self.illuminant_array is None:
                raise ValueError("illuminant_mode='array' ma illuminant_array=None.")
            E = np.asarray(self.illuminant_array, dtype=np.float64).reshape(-1)
            if E.size != L:
                raise ValueError(f"illuminant_array ha lunghezza {E.size}, atteso {L}.")
            if self.normalize_illuminant:
                E = E / (E.max() + 1e-12)
        else:
            raise ValueError("illuminant_mode deve essere 'planck' | 'none' | 'array'.")

        return E


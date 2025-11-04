import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SignalFlourFolderDataset(Dataset):
    """
    Legge HSI in .h5 organizzati per classi: root/class_name/*.h5

    Modalità:
      1) Proiezione HSI → RGB o RGB-IR usando:
         - curve spettrali dal CSV (wavelength, red, green, blue[, IR850])
         - illuminante (alogeno Planck 2856 K di default, oppure None o array custom)
         Output: (x, y) con x di shape (C, H, W), C=3 (RGB) o 4 (RGB-IR)

      2) patch_mean=True → solo media spettrale della patch (nessuna proiezione)
         Output: (x, y) con x di shape (1, L), tipicamente (1, 121)
    """

    # ---- Costanti utili ----
    DEFAULT_WAVELENGTHS = np.arange(400.0, 1000.0 + 1e-9, 5.0)  # 400..1000 step 5 → 121 canali

    def __init__(
        self,
        root: str,
        spectral_sens_csv: str,
        rgb: bool = True,
        ir: bool = False,
        dataset_keys="patch",
        exclude_prefixes=("._",),
        dtype=torch.float32,
        hsi_channels_first: bool = False,     # True se HSI salvato come (L,H,W), altrimenti (H,W,L)
        illuminant_mode: str = "planck",      # "planck" | "none" | "array"
        illuminant_T: float = 2856.0,         # K, usato se illuminant_mode="planck"
        illuminant_array: np.ndarray = None,  # shape (L,), usato se illuminant_mode="array"
        return_path: bool = False,
        patch_mean: bool = False,             # <-- NUOVO: se True, restituisce (1, L) media patch
    ):
        super().__init__()
        self.root = root
        self.spectral_sens_csv = spectral_sens_csv
        self.rgb = rgb
        self.ir = ir
        # accetta sia stringa singola che lista/tupla
        if isinstance(dataset_keys, (str, bytes)):
            self.dataset_keys = [dataset_keys]
        else:
            self.dataset_keys = list(dataset_keys)
        self.exclude_prefixes = exclude_prefixes
        self.dtype = dtype
        self.hsi_channels_first = hsi_channels_first
        self.illuminant_mode = illuminant_mode
        self.illuminant_T = illuminant_T
        self.illuminant_array = illuminant_array
        self.return_path = return_path
        self.patch_mean = patch_mean

        if not self.patch_mean and not (self.rgb or self.ir):
            raise ValueError("Imposta almeno uno tra rgb=True o ir=True, oppure patch_mean=True.")

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

        # Se siamo in modalità patch_mean, non servono curve/illuminante
        if not self.patch_mean:
            # ---- carica curve sensore dal CSV (una volta) ----
            self._sens_wl, self._sens_mat, self._sens_labels = self._load_sens_csv(self.spectral_sens_csv)
            # decide quali colonne usare
            self._use_cols = self._decide_channels(self.rgb, self.ir, self._sens_labels)
            self._out_channels = len(self._use_cols)
            # cache della matrice di proiezione per un dato vettore di λ
            self._cached_wl_key = None
            self._proj_matrix = None  # shape (L, C)

    # ---------- API Dataset ----------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        cube, wl = self._load_h5(path)          # cube: (H,W,L), wl: (L,)

        # Se il file non contiene wl, usa [400..1000] step 5
        if wl is None or wl.size == 0:
            wl = self.DEFAULT_WAVELENGTHS

        # --- Modalità 2: media spettrale patch ---
        if self.patch_mean:
            # media su H e W → (L,)
            spec = cube.reshape(-1, cube.shape[-1]).mean(axis=0).astype(np.float64)
            x = torch.from_numpy(spec[None, :]).to(self.dtype)  # (1, L)
            return (x, y, path) if self.return_path else (x, y)

        # --- Modalità 1: proiezione RGB / RGB-IR ---
        wl_key = (float(wl[0]), float(wl[-1]), int(wl.size))
        if (getattr(self, "_proj_matrix", None) is None) or (wl_key != getattr(self, "_cached_wl_key", None)):
            self._proj_matrix = self._build_projection(wl)   # (L, C)
            self._cached_wl_key = wl_key

        H, W, L = cube.shape
        X = cube.reshape(-1, L) @ self._proj_matrix   # (H*W, C)
        X = X.reshape(H, W, self._out_channels)
        X = np.moveaxis(X, -1, 0)                     # (C, H, W)
        x = torch.from_numpy(X).to(self.dtype)
        return (x, y, path) if self.return_path else (x, y)

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
            if k in f:
                return f[k][()]
        return None

    def _load_h5(self, path):
        import h5py, numpy as np

        with h5py.File(path, "r") as f:
            arr = None
            # 1) prova i dataset key in ordine
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

            # wavelengths (se esistono, prendi quelle)
            wl = None
            for k in ("wavelength", "wavelengths", "lambda", "bands"):
                if k in f and isinstance(f[k], h5py.Dataset):
                    wl = np.asarray(f[k][()], dtype=np.float64).reshape(-1)
                    break

        # --- coerzione forma ---
        if arr.ndim != 3:
            raise ValueError(f"atteso 3D, trovato {arr.shape} in {path}")

        # Se è (H,W,C) portalo a (C,H,W) se serve capire
        if arr.shape[0] not in (121, 3, 4) and arr.shape[-1] in (121, 3, 4):
            arr = np.moveaxis(arr, -1, 0)  # (C,H,W)

        # Porta a (H,W,L)
        if arr.shape[0] == 121:  # (C=121,H,W) → (H,W,L=121)
            cube = np.moveaxis(arr, 0, -1)
        elif arr.shape[0] in (3, 4):
            cube = np.moveaxis(arr, 0, -1)
        else:
            cube = np.moveaxis(arr, 0, -1)

        # wl di default se mancano e se è davvero spettrale (121 canali)
        if wl is None and cube.shape[-1] == 121:
            wl = np.arange(400.0, 1000.0 + 1e-9, 5.0)
        elif wl is None:
            wl = np.arange(cube.shape[-1], dtype=np.float64)

        return cube.astype(np.float64), wl

    @staticmethod
    def _load_sens_csv(csv_path):
        """
        CSV con header: wavelength, red, green, blue[, IR850]
        Ritorna: (wl_sens(Ls,), S_all(Ls,Nc), labels)
        """
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(",")
        hmap = {name.strip().lower(): i for i, name in enumerate(header)}
        if "wavelength" not in hmap:
            raise ValueError("Nel CSV manca la colonna 'wavelength'.")

        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        wl_sens = data[:, hmap["wavelength"]].astype(np.float64)

        labels = []
        cols_idx = []
        for name in ["red", "green", "blue", "ir850", "ir"]:
            if name in hmap:
                labels.append("IR850" if name in ("ir850", "ir") else name)  # normalizza label IR
                cols_idx.append(hmap[name])
        if not {"red", "green", "blue"}.issubset(set([l for l in labels if l != "IR850"])):
            raise ValueError("Nel CSV devono esserci almeno le colonne red, green, blue.")
        S_all = data[:, cols_idx].astype(np.float64)  # (Ls, Nc)
        return wl_sens, S_all, labels

    @staticmethod
    def _decide_channels(rgb, ir, labels):
        """Restituisce gli indici di colonna da usare (relativi a labels)."""
        name_to_idx = {n: i for i, n in enumerate(labels)}
        use = []
        if rgb:
            for n in ["red", "green", "blue"]:
                if n not in name_to_idx:
                    raise ValueError(f"Manca la colonna '{n}' nel CSV.")
                use.append(name_to_idx[n])
        if ir:
            if "IR850" not in labels:
                raise ValueError("Richiesto IR ma nel CSV non c'è 'IR850'.")
            use.append(name_to_idx["IR850"])
        return use

    def _build_projection(self, wl_target):
        """
        Costruisce la matrice (L, C) con integrazione su λ:
            proj(λ,c) = S_c(λ_interp) * E(λ) * Δλ
        dove S_c sono le curve (interpolate sulle wl_target), E è l'illuminante, Δλ pesi trapezoidali.
        """
        wl_target = np.asarray(wl_target, dtype=np.float64).reshape(-1)
        L = wl_target.size

        # Interpola curve sensore su wl_target
        S_interp_all = []
        for c in range(self._sens_mat.shape[1]):
            S_interp_all.append(np.interp(wl_target, self._sens_wl, self._sens_mat[:, c], left=0.0, right=0.0))
        S_interp_all = np.stack(S_interp_all, axis=1)  # (L, Nc_tot)
        S_interp = S_interp_all[:, self._use_cols]     # (L, C)

        # Illuminante
        if self.illuminant_mode == "none":
            E = np.ones(L, dtype=np.float64)
        elif self.illuminant_mode == "planck":
            E = self.planck_spd(wl_target, T=self.illuminant_T, normalize=True)
        elif self.illuminant_mode == "array":
            if self.illuminant_array is None:
                raise ValueError("illuminant_mode='array' ma illuminant_array=None.")
            E = np.asarray(self.illuminant_array, dtype=np.float64).reshape(-1)
            if E.size != L:
                raise ValueError(f"illuminant_array ha lunghezza {E.size}, atteso {L}.")
        else:
            raise ValueError("illuminant_mode deve essere 'planck' | 'none' | 'array'.")

        # Pesi Δλ (trapezi, non-uniforme OK)
        dl = np.zeros(L, dtype=np.float64)
        if L > 1:
            d = np.diff(wl_target)
            dl[1:-1] = 0.5 * (d[:-1] + d[1:])
            dl[0] = d[0] * 0.5
            dl[-1] = d[-1] * 0.5
        else:
            dl[:] = 1.0

        proj = (E[:, None] * S_interp) * dl[:, None]   # (L, C)
        return proj
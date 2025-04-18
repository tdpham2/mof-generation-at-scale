from dataclasses import dataclass
import subprocess
import os
from pathlib import Path
import shutil
import numpy as np
import ase

_file_dir = Path(__file__).parent / "files" / "raspa2_template"


@dataclass
class RASPA2Runner:
    """Interface for running pre-defined RASPA2 workflows."""

    raspa2_command: str = ""
    """Invocation used to run RASPA2 on this system"""

    run_dir: Path = Path("raspa2-runs")
    """Path to store RASPA2 files"""

    def _calculate_cell_size(
        self, atoms: ase.Atoms, cutoff: float = 12.8
    ) -> list[int, int, int]:
        """Method to calculate Unitcells (for periodic boundary condition) for GCMC

        Args:
            atoms (ase.Atoms): ASE atom object
            cutoff (float, optional): Cutoff in Angstrom. Defaults to 12.8.

        Returns:
            list[int, int, int]: Unit cell in x, y and z
        """
        unit_cell = atoms.cell[:]
        # Unit cell vectors
        a = unit_cell[0]
        b = unit_cell[1]
        c = unit_cell[2]
        # minimum distances between unit cell faces
        wa = np.divide(
            np.linalg.norm(np.dot(np.cross(b, c), a)),
            np.linalg.norm(np.cross(b, c)),
        )
        wb = np.divide(
            np.linalg.norm(np.dot(np.cross(c, a), b)),
            np.linalg.norm(np.cross(c, a)),
        )
        wc = np.divide(
            np.linalg.norm(np.dot(np.cross(a, b), c)),
            np.linalg.norm(np.cross(a, b)),
        )

        uc_x = int(np.ceil(cutoff / (0.5 * wa)))
        uc_y = int(np.ceil(cutoff / (0.5 * wb)))
        uc_z = int(np.ceil(cutoff / (0.5 * wc)))

        return [uc_x, uc_y, uc_z]

    def _write_cif(self, atoms: ase.Atoms, out_dir: str, name: str):
        """Save a CIF file with partial charges for RASPA2 from an ASE Atoms object.

        Args:
            atoms (ase.Atoms): ASE Atoms object containing partial charges in atoms.info["_atom_site_charge"].
            out_dir (str): Directory to save the output file.
            name (str): Name of the output file.
        """

        with open(os.path.join(out_dir, name), "w") as fp:
            fp.write(f"MOFA-{name}\n")

            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            fp.write(f"_cell_length_a      {a}\n")
            fp.write(f"_cell_length_b      {b}\n")
            fp.write(f"_cell_length_c      {c}\n")
            fp.write(f"_cell_angle_alpha   {alpha}\n")
            fp.write(f"_cell_angle_beta    {beta}\n")
            fp.write(f"_cell_angle_gamma   {gamma}\n")
            fp.write("\n")
            fp.write("_symmetry_space_group_name_H-M    'P 1'\n")
            fp.write("_symmetry_int_tables_number        1\n")
            fp.write("\n")

            fp.write("loop_\n")
            fp.write("  _symmetry_equiv_pos_as_xyz\n")
            fp.write("  'x, y, z'\n")
            fp.write("loop_\n")
            fp.write("  _atom_site_label\n")
            fp.write("  _atom_site_occupancy\n")
            fp.write("  _atom_site_fract_x\n")
            fp.write("  _atom_site_fract_y\n")
            fp.write("  _atom_site_fract_z\n")
            fp.write("  _atom_site_thermal_displace_type\n")
            fp.write("  _atom_site_B_iso_or_equiv\n")
            fp.write("  _atom_site_type_symbol\n")
            fp.write("  _atom_site_charge\n")

            coords = atoms.get_scaled_positions().tolist()
            symbols = atoms.get_chemical_symbols()
            occupancies = [
                1 for i in range(len(symbols))
            ]  # No partial occupancy
            charges = atoms.info["_atom_site_charge"]

            no = {}

            for symbol, pos, occ, charge in zip(
                symbols, coords, occupancies, charges
            ):
                if symbol in no:
                    no[symbol] += 1
                else:
                    no[symbol] = 1

                fp.write(
                    f"{symbol}{no[symbol]} {occ:.1f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} Biso 1.0 {symbol} {charge:.6f}\n"
                )

    def _get_cif_from_chargemol(
        self,
        cp2k_path: str,
        chargemol_fname: str = "DDEC6_even_tempered_net_atomic_charges.xyz",
    ) -> ase.Atoms:
        """Return an ASE atom object from a Chargemol output file.

        Args:
            cp2k_path (str): Path to Chargemol output.
            chargemol_fname (str, optional): Chargemol output filename. Defaults to "DDEC6_even_tempered_net_atomic_charges.xyz".

        Returns:
            ase.Atoms: ASE Atoms object containing partial charges in atoms.info["_atom_site_charge"].
        """
        with open(os.path.join(cp2k_path, chargemol_fname), "r") as f:
            symbols = []
            x = []
            y = []
            z = []
            charges = []
            positions = []
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i == 0:
                    natoms = int(line)
                elif i == 1:
                    data = line.split()
                    a1 = float(data[10])
                    a2 = float(data[11])
                    a3 = float(data[12])
                    b1 = float(data[15])
                    b2 = float(data[16])
                    b3 = float(data[17])
                    c1 = float(data[20])
                    c2 = float(data[21])
                    c3 = float(data[22])

                elif i <= natoms + 1:
                    data = line.split()
                    symbols.append(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    z = float(data[3])
                    charges.append(float(data[4]))
                    positions.append([x, y, z])
        cell = [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
        atoms = ase.Atoms(
            symbols=symbols, positions=positions, cell=cell, pbc=True
        )
        atoms.info["_atom_site_charge"] = charges

        return atoms

    def run_gcmc(
        self,
        name: str,
        cp2k_path: str,
        adsorbate: str,
        temperature: float,
        pressure: float,
        cutoff: float = 12.8,
        n_cycle: int = 500,
        chargemol_fname: str = "DDEC6_even_tempered_net_atomic_charges.xyz",
    ) -> tuple[float, float, float, float]:
        """Execute RASPA2 with specified input parameters.

        Args:
            name (str): Name of the MOF (excluding the ".cif" extension).
            cp2k_path (str): Path to the Chargemol output file.
            adsorbate (str): Adsorbate name. Supported values: 'H2', 'CO2', 'N2'.
            temperature (float): Simulation temperature in Kelvin (K).
            pressure (float): Simulation pressure in Pascal (Pa).
            cutoff (float, optional): Van der Waals and Coulomb cutoff distance in Angstroms. Defaults to 12.8 Å.
            n_cycle (int, optional): Number of initialization and production cycles. Defaults to 500.
            chargemol_fname (str, optional): Filename of the Chargemol output. Defaults to "DDEC6_even_tempered_net_atomic_charges.xyz".

        Returns:
            tuple[float, float, float, float]: Computed uptake (U) and error (E) from RASPA2 in the following order:
                - U (mol/kg)
                - E (mol/kg)
                - U (g/L)
                - E (g/L)
        """

        out_dir = (
            self.run_dir / f"{name}_{adsorbate}_{temperature}_{pressure:0e}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        atoms = self._get_cif_from_chargemol(
            cp2k_path, chargemol_fname=chargemol_fname
        )
        # Write CIF file with charges
        self._write_cif(atoms, out_dir=out_dir, name=name + ".cif")

        # Copy other input files (simulation.input, force fields and definition files) from template folder.
        subprocess.run(f"cp {_file_dir}/* {out_dir}/", shell=True)

        [uc_x, uc_y, uc_z] = self._calculate_cell_size(atoms=atoms)

        # Modify input from template simulation.input
        with (
            open(f"{out_dir}/simulation.input", "r") as f_in,
            open(f"{out_dir}/simulation.input.tmp", "w") as f_out,
        ):
            for line in f_in:
                if "NCYCLE" in line:
                    line = line.replace("NCYCLE", str(n_cycle))
                if "ADSORBATE" in line:
                    line = line.replace("ADSORBATE", adsorbate)
                if "TEMPERATURE" in line:
                    line = line.replace("TEMPERATURE", str(temperature))
                if "PRESSURE" in line:
                    line = line.replace("PRESSURE", str(pressure))
                if "UC_X UC_Y UC_Z" in line:
                    line = line.replace(
                        "UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}"
                    )
                if "CUTOFF" in line:
                    line = line.replace("CUTOFF", str(cutoff))
                if "CIFFILE" in line:
                    line = line.replace("CIFFILE", name)
                f_out.write(line)

        shutil.move(
            f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input"
        )

        # Run RASPA2
        subprocess.run(self.raspa2_command, shell=True, cwd=out_dir)

        # Get output from Output/ folder
        system_dir = os.path.join(out_dir, "Output", "System_0")
        output_file = next(
            (
                f
                for f in os.listdir(system_dir)
                if f.startswith("output_") and f.endswith(".data")
            ),
            None,
        )
        if not output_file:
            raise FileNotFoundError(
                "No output_*.data file found in Output/System_0"
            )
        output_path = os.path.join(system_dir, output_file)
        mol_kg_line = mg_g_line = density_line = None
        with open(output_path, "r") as file:
            for line in file:
                if "Average loading absolute [mol/kg framework]" in line:
                    mol_kg_line = line.strip()
                elif (
                    "Average loading absolute [milligram/gram framework]"
                    in line
                ):
                    mg_g_line = line.strip()
                elif "Framework Density" in line:
                    density_line = line.strip()

        if not all([mol_kg_line, mg_g_line, density_line]):
            raise ValueError("One or more expected lines were not found.")

        uptake_mol_kg = float(mol_kg_line.split()[5])
        error_mol_kg = float(mol_kg_line.split()[7])

        density_kg_m3 = float(density_line.split()[2])
        uptake_mg_g = float(mg_g_line.split()[5])
        error_mg_g = float(mg_g_line.split()[7])

        # Unit conversion to g/L
        uptake_g_L = uptake_mg_g * density_kg_m3 / 1000
        error_g_L = error_mg_g * density_kg_m3 / 1000
        return uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L

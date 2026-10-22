import argparse, csv, os, glob
import pyvista as pv
import numpy as np

VTK_CELL_TYPES = {
    "tri3":  (3, 5),   
    "tri6":  (6, 22),  
    "quad4": (4, 9),   
    "quad8": (8, 23),
}
# handelling functions 
def write_vtk(points, elems, elem_type, point_data=None, cell_data=None, out_vtk="output.vtk"):
    """
    Write an unstructured mesh of various types to VTK with optional nodal and cell data.

    Parameters
    ----------
    points : (N, 3) array
        Coordinates of the nodes.
    elems : (M, n_nodes) array
        Connectivity of elements.
    elem_type : str
        Element type: "tri3", "tri6", "quad4", "quad8".
    point_data : dict[str, np.ndarray], optional
        Nodal fields; each array must have length N.
    cell_data : dict[str, np.ndarray], optional
        Element fields; each array must have length M.
    out_vtk : str
        Output file path.
    """
    points = np.asarray(points)
    elems = np.asarray(elems, dtype=int)

    if elem_type not in VTK_CELL_TYPES:
        raise ValueError(f"Unknown element type: {elem_type}")

    n_nodes, vtk_cell_type = VTK_CELL_TYPES[elem_type]
    n_elems = elems.shape[0]

    if elems.shape[1] != n_nodes:
        raise ValueError(f"Expected {n_nodes} nodes/element for {elem_type}, got {elems.shape[1]}")

    # Build VTK cell array
    cell_array = np.hstack([np.full((n_elems, 1), n_nodes, dtype=np.int64), elems]).flatten()
    cell_types = np.full(n_elems, vtk_cell_type, dtype=np.uint8)

    # Create the unstructured grid
    grid = pv.UnstructuredGrid(cell_array, cell_types, points)

    # Add point fields
    if point_data:
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            if len(arr) != len(points):
                raise ValueError(f"Nodal field '{name}' length {len(arr)} != number of points {len(points)}")
            grid.point_data[name] = arr

    # Add cell fields
    if cell_data:
        for name, arr in cell_data.items():
            arr = np.asarray(arr)
            if len(arr) != n_elems:
                raise ValueError(f"Cell field '{name}' length {len(arr)} != number of elements {n_elems}")
            grid.cell_data[name] = arr

    # Write file
    grid.save(out_vtk, binary=False)
    print(f"- Writting information from an input file into a VTK format :{out_vtk}!")
def von_mises_3d(sigma):
    """
    Compute 3D von Mises stress from Cauchy stress tensor(s).
    Input:
      sigma : array-like shape (3,3) or (...,3,3)
              symmetric stress tensor(s)
    Returns:
      scalar or array of von Mises values
    """
    s = np.asarray(sigma)
    single = False
    if s.ndim == 2:
        s = s[np.newaxis, ...]
        single = True

    sxx = s[..., 0, 0]
    syy = s[..., 1, 1]
    szz = s[..., 2, 2]
    sxy = 0.5 * (s[..., 0, 1] + s[..., 1, 0])
    syz = 0.5 * (s[..., 1, 2] + s[..., 2, 1])
    szx = 0.5 * (s[..., 2, 0] + s[..., 0, 2])

    part1 = 0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
    part2 = 3.0 * (sxy**2 + syz**2 + szx**2)
    vm = np.sqrt(part1 + part2)

    return vm[0] if single else vm
# def area_tri3(pts, conn):
#     a, b, c = np.asarray(pts[conn[0]-1]), np.asarray(pts[conn[1]-1]), np.asarray(pts[conn[2]-1])
#     return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
# def area_tri6(pts, conn):
#     # first three are corner nodes
#     a, b, c = np.asarray(pts[conn[0]-1]), np.asarray(pts[conn[1]-1]), np.asarray(pts[conn[2]-1])
#     return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
# def area_quad4(pts, conn):
#     p0, p1, p2, p3 = np.asarray(pts[conn[0]-1]), np.asarray(pts[conn[1]-1]), np.asarray(pts[conn[2]-1]), np.asarray(pts[conn[3]-1])
#     area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
#     area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))
#     return area1 + area2
# def area_quad8(pts, conn):
#     p0, p1, p2, p3 = np.asarray(pts[conn[0]-1]), np.asarray(pts[conn[1]-1]), np.asarray(pts[conn[2]-1]), np.asarray(pts[conn[3]-1])
#     area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
#     area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))
#     return area1 + area2
def area_tri3(pts, conn):
    a = np.asarray(pts[conn[0]])
    b = np.asarray(pts[conn[1]])
    c = np.asarray(pts[conn[2]])
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def area_tri6(pts, conn):
    a = np.asarray(pts[conn[0]])
    b = np.asarray(pts[conn[1]])
    c = np.asarray(pts[conn[2]])
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def area_quad4(pts, conn):
    p0 = np.asarray(pts[conn[0]])
    p1 = np.asarray(pts[conn[1]])
    p2 = np.asarray(pts[conn[2]])
    p3 = np.asarray(pts[conn[3]])

    area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))

    return area1 + area2

def area_quad8(pts, conn):
    p0 = np.asarray(pts[conn[0]])
    p1 = np.asarray(pts[conn[1]])
    p2 = np.asarray(pts[conn[2]])
    p3 = np.asarray(pts[conn[3]])

    area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))

    return area1 + area2
def element_area(pts, conn, cell_type):
    if cell_type in ("tri3", 5):     
        return area_tri3(pts, conn)
    elif cell_type in ("tri6", 22):
        return area_tri6(pts, conn)
    elif cell_type in ("quad4", 9):
        return area_quad4(pts, conn)
    elif cell_type in ("quad8", 23):
        return area_quad8(pts, conn)
    else:
        raise ValueError(f"Unsupported element type: {cell_type}")
def local_basis_from_element(pts, elem):
    """Compute local coordinate basis (t1, t2, n) for a shell element."""
    p0 = np.array(pts[elem[0]])
    p1 = np.array(pts[elem[1]])
    p2 = np.array(pts[elem[2]])

    # Tangent directions
    t1 = p1 - p0
    t1 /= np.linalg.norm(t1)
    v2 = p2 - p0

    n = np.cross(t1, v2)
    if np.linalg.norm(n) < 1e-12:  # avoid degenerate element normals
        n = np.array([0.0, 0.0, 1.0])
    else:
        n /= np.linalg.norm(n)

    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)

    R = np.vstack((t1, t2, n)).T  # local → global
    return R, t1, t2, n
def transform_to_local(stress_global, R):
    """
    Transform a 3x3 stress tensor from global to local coordinates.
    """
    return R.T @ stress_global @ R
def in_plane_stress(stress_local):
    """
    Extract 2x2 in-plane stress tensor from local 3x3 tensor.
    """
    return stress_local[:2, :2]
def von_mises_2d(stress_2x2):
    """Compute 2D von Mises stress from in-plane tensor."""
    sxx, syy = stress_2x2[0, 0], stress_2x2[1, 1]
    sxy = stress_2x2[0, 1]
    return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

def sym_tensor(tensor):
    """Return the symmetric part of a 3x3 tensor."""
    tensor = np.asarray(tensor, dtype=float).reshape(3, 3)
    return 0.5 * (tensor + tensor.T)

def make_table(caseID,time,nelem,npts,cell_type,all_mask,von_mises,inplane_von_mises,
               eig_ratios,unidir,areas,thickness,
               output_csv = "batch_results.csv"):
    all_data = []
    for name, mask in all_mask.items():
        # extracted aneurysm data
        inplane_von_mises_region = []
        von_mises_region = []
        thickness_region = []
        area_region = []
        eig_ratios_region = []
        unidir_region = []

        for ele in range(nelem):
            if mask[ele] == 1 :
                von_mises_region.append(float(von_mises[ele]))
                inplane_von_mises_region.append(float(inplane_von_mises[ele]))
                eig_ratios_region.append(float(eig_ratios[ele]))
                unidir_region.append(int(unidir[ele]))
                area_region.append(float(areas[ele]))
                thickness_region.append(float(thickness[ele]))

        # von Mises data
        if von_mises_region and area_region:
            von_mises_weighted_mean = np.average(von_mises_region, weights=area_region)
            # print(f"* weighted average von-Mises stress in aneurysm region: "
                # f"{von_mises_weighted_mean:.2e} "
                # f"(max: {np.max(von_mises_region):.2e}, "
                # f"min: {np.min(von_mises_region):.2e})")
        else:
            von_mises_weighted_mean = np.nan
            raise ValueError("* Error: No data found in von_mises field at mask region!")
            
        # local von Mises data
        if inplane_von_mises_region and area_region:
            inplane_von_mises_weighted_mean = np.average(inplane_von_mises_region, weights=area_region)
            # print(f"* weighted average local von-Mises stress in aneurysm region: "
            #     f"{inplane_von_mises_weighted_mean:.2e} "
            #     f"(max: {np.max(inplane_von_mises_aneu):.2e}, "
            #     f"min: {np.min(inplane_von_mises_aneu):.2e})")
        else:
            inplane_von_mises_weighted_mean = np.nan
            raise ValueError("* Error: No data found in inplane_von_mises field at mask region!")
        
        # Ratio Eigenvalues data
        if eig_ratios_region and area_region:
            eig_ratios_weighted_mean = np.average(eig_ratios_region, weights=area_region)
            # print(f"* weighted average Ratio Eigenvalues in aneurysm region: "
            #     f"{eig_ratios_weighted_mean:.2e} "
            #     f"(max: {np.max(eig_ratios_aneu):.2e}, "
            #     f"min: {np.min(eig_ratios_aneu):.2e})")
        else:
            eig_ratios_weighted_mean = np.nan
            raise ValueError("* Error: No data found in Ratio Eigenvalues field at mask region!")
        # Unidirectional area aneurysm data
        if unidir_region and area_region:
            unidir_weighted_mean = np.average(unidir_region, weights=area_region)
            # print(f"* Area of Unidirectional (r<0.1) in aneurysm region: "
            #     f"{unidir_1_weighted_mean:.2e} ")
        else:
            unidir_weighted_mean = np.nan
            raise ValueError("* Error: No data found in Unidirectional field at mask region!")

            
        # High-stress von Mises and stress concentration.
        # The high-stress region is defined by the upper quartile of von Mises
        # stress itself (NOT stress x area). The reported high-stress value is
        # the area-weighted mean within that region.
        if von_mises_region and area_region:
            stress = np.asarray(von_mises_region, dtype=float)
            area = np.asarray(area_region, dtype=float)

            q75 = np.percentile(stress, 75.0)
            high_mask = stress >= q75

            if np.any(high_mask) and np.sum(area[high_mask]) > 0:
                high_stress_von = np.average(stress[high_mask], weights=area[high_mask])
                stress_concentration = high_stress_von / von_mises_weighted_mean
            else:
                high_stress_von = np.nan
                stress_concentration = np.nan
        else:
            high_stress_von = np.nan
            stress_concentration = np.nan
            raise ValueError("* Error: No data found for high-stress von Mises calculation!")

        # Secondary in-plane concentration metrics, using the upper quartile
        # of the in-plane von Mises stress itself.
        if inplane_von_mises_region and area_region and eig_ratios_region:
            inplane_stress = np.asarray(inplane_von_mises_region, dtype=float)
            inplane_area = np.asarray(area_region, dtype=float)
            eig_region = np.asarray(eig_ratios_region, dtype=float)

            threshold = np.percentile(inplane_stress, 75.0)
            high_ip = inplane_stress >= threshold
            upper_inplane_von = inplane_stress[high_ip]
            upper_eig_ratios = eig_region[high_ip]
            upper_area = inplane_area[high_ip]

            if upper_area.sum() > 0:
                high_inplane_von = np.average(upper_inplane_von, weights=upper_area)
                upper_quartile_avg_inplane_von = high_inplane_von / inplane_von_mises_weighted_mean
                high_eig_ratio = np.average(upper_eig_ratios, weights=upper_area)
                upper_quartile_avg_eig_ratios = high_eig_ratio / eig_ratios_weighted_mean if abs(eig_ratios_weighted_mean) > 1e-30 else np.nan
            else:
                upper_quartile_avg_inplane_von = np.nan
                upper_quartile_avg_eig_ratios = np.nan

            # print(f"* upper quartile inplane stress concentration: {upper_quartile_avg_inplane_von:.2e}")
            # print(f"* upper quartile eigen ratios concentration: {upper_quartile_avg_eig_ratios:.2e}")
        else:
            upper_quartile_avg_inplane_von = np.nan
            upper_quartile_avg_eig_ratios = np.nan
            raise ValueError("* Error: No data found in inplane_von_mises_region field at mask region and make problem for calculating inplane concentration of stress!")
            
        # aspect aneurysm data
        aspect = []
        for ele in range(len(area_region)):
            area_val = area_region[ele]
            if area_val > 1e-18:
                aspect.append(float(thickness_region[ele] / np.sqrt(area_val)))  
            else:
                aspect.append(np.nan)
        # if aspect:
        #     print(f"* average aspect ratio of shell element in aneurysm region: "
        #         f"{np.nanmean(aspect):.2e} "
        #         f"(max: {np.nanmax(aspect):.2e}, "
        #         f"min: {np.nanmin(aspect):.2e})")

        # output dictionary
        output = {
        'caseID': caseID,
        'time': time,
        'mesh_type': cell_type,
        'nelem': nelem,
        'npts': npts,
        'mask': name,
        'area': np.sum(area_region),
        'mean_aspect': np.mean(aspect),
        'max_aspect': np.max(aspect),
        'min_aspect': np.min(aspect),
        'mean_von': von_mises_weighted_mean,
        'high_stress_von': high_stress_von,
        'max_von': np.max(von_mises_region),
        'min_von': np.min(von_mises_region),
        "mean_inplane_vm": inplane_von_mises_weighted_mean,
        "max_inplane_vm": np.max(inplane_von_mises_region),
        "min_inplane_vm": np.min(inplane_von_mises_region),
        'mean_eigens_ratio':eig_ratios_weighted_mean,
        'mean_unidir': unidir_weighted_mean,
        'concen_von': stress_concentration,
        'concen_inplane_von': upper_quartile_avg_inplane_von,
        'concen_eig_ratios': upper_quartile_avg_eig_ratios
        }
        all_data.append(output)
    # save results
    fieldnames=["caseID", "time", "mesh_type", "nelem", "npts","mask","area",
                                      "mean_aspect","max_aspect","min_aspect", 
                                      "mean_von", "high_stress_von", "max_von","min_von",
                                      "mean_inplane_vm","max_inplane_vm","min_inplane_vm",
                                      "mean_eigens_ratio","mean_unidir",
                                      "concen_von","concen_inplane_von","concen_eig_ratios"]
    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0

    if file_exists:
        with open(output_csv, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header != fieldnames:
            raise ValueError(
                f"Existing CSV '{output_csv}' has a different schema. "
                "Do not append revised results to an older CSV. "
                "Delete/rename the old file or use -n with a new filename."
            )

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for data in all_data:
            writer.writerow(data)

    print(f"- All results saved to {output_csv}")

pa = argparse.ArgumentParser (description="Importing the FEBio-VTK exported files, and mask file to extract mechanical stress features within the masked regions for statistical analysis.")
pa.add_argument('vtk_inputs',type=str, nargs="+",metavar="FILE", help='FEBio exported files in vtk format @ specific time steps')
pa.add_argument('reg_file', help='zfem file with region field')
pa.add_argument('caseid' ,help='the ID of the case which using in the output result')
pa.add_argument('--mask','-m',dest='mask_file', type=str, nargs="+", metavar="FILE", help='txt file with with binary value to shows a specific region in domain')
pa.add_argument('--time','-t',dest='run_time',type=float,default=1.0,
                help='target FEBio output time (default: 1.0; end of prestress Step 1)')
pa.add_argument('--name','-n',dest='name',type=str,help='name of the output')
pa.add_argument('--area', '-an', action='store_true', help='covering area for all element 1')

arg=pa.parse_args()

reg_file=arg.reg_file
caseID=arg.caseid
if arg.name:
    output_csv = arg.name
else:
    output_csv = 'batch_results_stress_metrics.csv'

vtk_inputs_no_checked = []
for inp in arg.vtk_inputs:
    if "x" in inp:
        pattern = inp.replace("x", "*")
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise ValueError(f"* No files match pattern: {inp}")
        vtk_inputs_no_checked.extend(matched)
    else:
        if not os.path.isfile(inp):
            raise ValueError(f"* File not found: {inp}")
        vtk_inputs_no_checked.append(inp)
vtk_inputs_no_checked = sorted(set(vtk_inputs_no_checked))
print(f"- the {len(vtk_inputs_no_checked)} VTK files used!")
vtk_inputs = []

def read_vtk_time(path):
    """Read FEBio time from the VTK title line, e.g. 'time 1'."""
    with open(path, 'r') as f:
        # FEBio ASCII VTK files place the time in the title/header line.
        # Search only the first few lines to avoid matching field names later.
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            values = line.strip().split()
            if len(values) >= 2 and values[0].lower() == 'time':
                try:
                    return float(values[1])
                except ValueError:
                    raise ValueError(
                        f"Invalid FEBio time value in {path}: {line.strip()}"
                    )
    raise ValueError(f"Could not read FEBio time from {path}")

target_time = float(arg.run_time)
time_tol = 1e-6
candidates = []
for inp in vtk_inputs_no_checked:
    try:
        readed_time = read_vtk_time(inp)
    except Exception as exc:
        print(f"* WARNING: {exc}")
        continue
    if readed_time >= target_time - time_tol:
        candidates.append((abs(readed_time - target_time), readed_time, inp))
    print(f"*** time read from {inp}: {readed_time:.10g}; target: {target_time:.10g}")

if not candidates:
    raise ValueError("* ERROR: no readable VTK file found!")

candidates.sort(key=lambda item: (item[0], item[1]))
best_diff, best_time, best_file = candidates[0]
if best_diff > time_tol:
    if best_time < target_time:
        raise ValueError(
            f"* ERROR: no VTK file at requested time {target_time}. "
            f"Closest is {best_time} in {best_file}."
        )

vtk_inputs.append(best_file)
print(f"- Selected VTK at time {best_time}: {best_file}")

mask_file = []
for inp in (arg.mask_file or []):
    if "x" in inp:
        pattern = inp.replace("x", "*")
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise ValueError(f"* No files match pattern: {inp}")
        mask_file.extend(matched)
    else:
        if not os.path.isfile(inp):
            raise ValueError(f"* File not found: {inp}")
        mask_file.append(inp)

mask_file = sorted(set(mask_file))
print(f"- Using {len(mask_file)} mask files: ",mask_file)
# check the format of inported files:
for vtk_input in vtk_inputs:
    if not vtk_input.endswith('.vtk'):
        raise ValueError(f"* Error: the format of {vtk_input} is not vtk!")
if not reg_file.endswith('.zfem'):
    raise ValueError(f"* Error: the format of {reg_file} is not zfem!")
# check csv output file available in the root:
# if os.path.isfile(output_csv):
#     raise ValueError (f"* Error: {output_csv} file already exist in the root path!")
# read and check vtk file 
required_pointal_data={'shell_displacement', 'shell_director','shell_thickness'}
required_cell_data={'shell_strain', 'shell_top_strain', 'shell_bottom_strain', 'shell_top_stress', 'shell_bottom_stress', 'stress', 'PK2_stress'}
for vtk_input in vtk_inputs:
    time = read_vtk_time(vtk_input)

    print ("-"*10+f"start reading {vtk_input}"+"-"*10)
    mesh = pv.read(vtk_input)
    print(f"- Successfully loaded {vtk_input}")
    print(f"- Number of points: {mesh.n_points}")
    print(f"- Number of cells : {mesh.n_cells}")
    cells_types = []
    for ID in np.unique(mesh.celltypes):
        cell_type_name = None
        for name, (_, cell_ID) in VTK_CELL_TYPES.items():
            if cell_ID == ID:
                cell_type_name = name
                break

        if cell_type_name is None:
            raise ValueError(f"* Error: Unknown cell type ID {ID}")

        cells_types.append(cell_type_name)
    heterogeneity = "Homogeneous" if len(cells_types) == 1 else "Heterogeneous"
    print(f"- Cells type: {heterogeneity} - {cells_types}")
    if len(cells_types) != 1:
        raise ValueError(f"* Error: The type of cells are not uniformed, list of cell types: {cells_types} ")
    # check required field 
    for field in required_pointal_data:
        if field not in list(mesh.point_data.keys()):
            raise ValueError(f"* Error: field {field} does not avaiable in {vtk_input}")
    for field in required_cell_data:
        if field not in list(mesh.cell_data.keys()):
            raise ValueError(f"* Error: field {field} does not avaiable in {vtk_input}")
    print("- All required fields are avaiable")
    # read region file
    region_pts, region_cells,thickness_cells, elems = [], [], [], []
    npts, nelem = 0, 0
    if os.path.exists(reg_file):

        try:
            with open(reg_file, "r") as file:
                lines = [line.strip() for line in file]
                find_elems = False
                for i, line in enumerate(lines):
                    if line.upper().startswith("TRIANGLE"):
                        nelem = int(lines[i + 2].split()[0])
                        find_elems = True
                        if nelem != mesh.n_cells:
                            raise ValueError(
                                f"* Error: number of element in vtk ({mesh.n_cells}) "
                                f"and region file ({nelem}) does not match!")
                    if line.upper().startswith("POINTS"):
                        npts = int(lines[i + 1].split()[0])
                    if find_elems:
                        values = lines[i + 3].split()
                        if len(values) == 3:
                            elems.append([int(value) for value in values])
                        else:
                            raise ValueError(f"* Error: the cell type of {reg_file} is not tri3!")
                        if len(elems) == nelem:
                            find_elems = False
                            break
                if len(elems) != nelem:
                    raise ValueError(f"* Cannot read the ELEMENTS in {reg_file} properly!")

        except Exception as e:
            print("Error:", e)
        print(f"- number of points {npts} and element {nelem} in {reg_file}!")
        try:
            with open(reg_file, "r") as file:

                lines = [line.strip() for line in file]

                find_region = False

                for i, line in enumerate(lines):

                    if line.startswith("regions"):
                        find_region = True
                        continue

                    if find_region:
                        values = lines[i + 1].split()

                        if len(values) != 1:
                            raise ValueError(
                                f"* Error: number of components in line {i+1} "
                                f"of region field is more than 1!"
                            )

                        region_pts.append(int(values[0]))

                    if len(region_pts) == npts:
                        find_region = False
                        break

        except Exception as e:
            print("Error:", e)

        if len(region_pts) != npts:
            raise ValueError("* Error: code cannot read region field properly!")
        else:
            print(f"- Successfully read {reg_file} file!")
        # Convert the PointalData region to celldata region:
        for elem in elems:
            region_cells.append(region_pts[elem[0]-1])

    else:
        print(f"* {reg_file} does not exist!")
        try:
            with open(f"../data/{caseID}.flds.zfem", "r") as file:
                lines = [line.strip() for line in file]
                find_elems = False
                for i, line in enumerate(lines):
                    if line.upper().startswith("TRIANGLE"):
                        nelem = int(lines[i + 2].split()[0])
                        find_elems = True
                        if nelem != mesh.n_cells:
                            raise ValueError(
                                f"* Error: number of element in vtk ({mesh.n_cells}) "
                                f"and region file ({nelem}) does not match!")
                    if line.upper().startswith("POINTS"):
                        npts = int(lines[i + 1].split()[0])
                    if find_elems:
                        values = lines[i + 3].split()
                        if len(values) == 3:
                            elems.append([int(value) for value in values])
                        else:
                            raise ValueError(f"* Error: the cell type of ../data/{caseID}.flds.zfem is not tri3!")
                        if len(elems) == nelem:
                            find_elems = False
                            break
                if len(elems) != nelem:
                    raise ValueError(f"* Cannot read the ELEMENTS in ../data/{caseID}.flds.zfem properly!")

        except Exception as e:
            print("Error:", e)
        print(f"- number of points {npts} and element {nelem} in ../data/{caseID}.flds.zfem!")
        try:
            with open(f"../data/{caseID}.aneu.0.0", "r") as file:
                lines = [line.strip() for line in file]

            for i, line in enumerate(lines):

                values = line.split()

                if len(values) != 1:
                    raise ValueError(
                        f"* Error: number of components in line {i} "
                        f"of aneu.0.0 mask is more than 1!"
                    )

                region_cells.append(int(values[0]))

        except Exception as e:
            print("Error:", e)

    # convert pyVista DataStructure to list
    vtk_pnts, vtk_cells = [], []
    vtk_pnts=mesh.points.tolist()
    for i in range(mesh.n_cells):
        vtk_cells.append(list(mesh.get_cell(i).point_ids))
    # Convert nodal shell thickness to one mean thickness per VTK cell.
    # For Tri6, use all six shell nodes.
    for conn in vtk_cells:
        thickness_cells.append(
            float(np.mean(mesh.point_data['shell_thickness'][np.asarray(conn, dtype=int)]))
        )
    # calculate inplane stress
    stress_inplane_all = []
    t1_all, t2_all, n_all = [], [], []
    print("- Start Calculating the local coordinate system ...")
    for e, elem in enumerate(vtk_cells):
        # handle mid-node elements
        if len(elem) >= 6:
            # VTK quadratic triangle ordering: first three nodes are corners,
            # followed by the three midside nodes.
            elem_nodes = [elem[0], elem[1], elem[2]]
        else:
            elem_nodes = elem[:3]

        R, t1, t2, n = local_basis_from_element(vtk_pnts, elem_nodes)
        t1_all.append(t1)
        t2_all.append(t2)
        n_all.append(n)

        # transform global stress tensor to local
        sigma_global = np.array(mesh.cell_data['stress'][e]).reshape(3, 3)
        sigma_local = R.T @ sigma_global @ R
        sigma_inplane = sigma_local[:2, :2]

        stress_inplane_all.append(sigma_inplane)
    # calculate the area of each cells 
    areas = []
    if arg.area:
        for i in range(len(vtk_cells)):
            areas.append('1')
    else:
        for i, conn in enumerate(vtk_cells):
            areas.append(element_area(vtk_pnts, conn, cells_types[0]))
    areas = np.array(areas, dtype=float)


    # Compute in-plane von Mises
    inplane_von_mises_mid = [von_mises_2d(t) for t in stress_inplane_all]

    # Compute von Mises stress for each tensor
    von_mises_mid = []
    for tensor in mesh.cell_data['stress']:
        stress3x3 = np.array(tensor).reshape(3, 3)
        vm = von_mises_3d(stress3x3)
        von_mises_mid.append(float(vm))

    # Compute eigen values and r 
    stress_inplane_all = [np.array(t) for t in stress_inplane_all]

    max_eigs = []
    min_eigs = []
    eig_ratios = []

    for t in stress_inplane_all:
        # In-plane principal stresses. Use the symmetric tensor and order the
        # principal-stress magnitudes so that 0 <= r <= 1.
        t_sym = 0.5 * (t + t.T)
        eigvals = np.linalg.eigvalsh(t_sym)
        eig_magnitudes = np.sort(np.abs(eigvals))

        eig_min = eig_magnitudes[0]
        eig_max = eig_magnitudes[-1]
        max_eigs.append(eig_max)
        min_eigs.append(eig_min)

        if eig_max > 1e-12:
            eig_ratios.append(eig_min / eig_max)
        else:
            eig_ratios.append(0.0)

    max_eigs = np.array(max_eigs)
    min_eigs = np.array(min_eigs)
    eig_ratios = np.array(eig_ratios)

    # Uni-directional Stress area
    unidir_1 = (eig_ratios < 0.1).astype(int)
    unidir_5 = (eig_ratios < 0.5).astype(int)

    # creat table of for paramteres 
    mask={}
    if mask_file:
        for file in mask_file:
            data = []
            try:
                with open (file,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        values = line.strip().split()
                        if len(values) == 1:    
                            data.append(int(line))
                        else:
                            raise ValueError (f"* Error: number of componand in line {line} of file {file} is not ONE!")
            except FileExistsError:
                print(f"* Error: does not find {file}!")
            if len(np.unique(data)) != 2:
                raise ValueError (f"* Error: the mask file {file} is not binary field!")
            if len (data) != mesh.n_cells:
                raise ValueError (f"* Error: number of cell in mask file {file} ({len(data)}) no match with nelem of vtk file ({mesh.n_cells})!")
            mask[file]=data
    else:
        data = []
        for ele in range (mesh.n_cells):
            if region_cells[ele] in [4,8,16]:    
                data.append(int(1))
            else:
                data.append(int(0))
        mask['aneu']=data
    # check input and calculated fields
    cell_data, point_data = {}, {}
    for field in required_pointal_data:
        if field in list(mesh.point_data.keys()):
            point_data[field]=mesh.point_data[field]
    cell_data['region_cells'] = region_cells
    cell_data['area'] = areas
    cell_data['t1'] = t1_all
    cell_data['t2'] = t2_all
    cell_data['n'] = n_all
    cell_data['stress_inplane'] = stress_inplane_all
    cell_data['inplane_von_mises_mid'] = inplane_von_mises_mid
    cell_data['von_mises_mid'] = von_mises_mid
    cell_data['max_eigs'] = max_eigs
    cell_data['min_eigs'] = min_eigs
    cell_data['eig_ratios'] = eig_ratios
    cell_data['unidir_1'] = unidir_1
    cell_data['thickness_cells'] = thickness_cells
    for field, data in mask.items():
        cell_data[field] = data
    for field in required_cell_data:
        if field in list(mesh.cell_data.keys()):
            cell_data[field]=mesh.cell_data[field]
    write_vtk(vtk_pnts,vtk_cells,cells_types[0],cell_data=cell_data, point_data=point_data, out_vtk=f"{caseID}_fields.vtk")
    make_table(caseID,time,mesh.n_cells,mesh.n_points,cells_types[0],mask,
               von_mises_mid,inplane_von_mises_mid,eig_ratios,unidir_1,
               areas,thickness_cells,
               output_csv)
print ("-"*10+f"All cases readed an features calculated and save in csv file"+"-"*10)
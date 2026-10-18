# Python API

## Opening netCDF files

```{eval-rst}
.. autofunction:: gmpas.open_mpas
.. autofunction:: gmpas.open_data
.. autoclass:: gmpas.MpasMesh
   :members:
.. autoclass:: gmpas.MpasAccessor
   :members:
```

## Viewer

```{eval-rst}
.. autofunction:: gmpas.cell_field
.. autofunction:: gmpas.edge_field
.. autofunction:: gmpas.vectors
.. autofunction:: gmpas.mesh_structure
.. autofunction:: gmpas.save_figure
```

## Geometry and sampling

```{eval-rst}
.. autofunction:: gmpas.rasterize
.. autofunction:: gmpas.target_grid
.. autofunction:: gmpas.should_raster
.. autofunction:: gmpas.reconstruct_cell_winds
```

## Data operations

```{eval-rst}
.. autofunction:: gmpas.select
.. autofunction:: gmpas.spatial_dim
.. autofunction:: gmpas.field_label
.. autofunction:: gmpas.plottable
.. autofunction:: gmpas.has_mesh
```

## Configuration

```{eval-rst}
.. autoclass:: gmpas.Style
.. autofunction:: gmpas.resolve_extent
.. autofunction:: gmpas.cache_dir
.. autofunction:: gmpas.resolve_path
```

"""Prevent topology-cache corruption and verify non-unit Tmf derivatives."""
import numpy as np
import pytest
from scipy import sparse

from reservoir_backend.comp.dual_residual import dual_residual, pack_dual, unpack_dual
from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_blocks import _csc_cached, assemble_block_jacobian
from reservoir_backend.solver.dpdp_context import DPDPModelContext


def test_csc_cache_distinguishes_equal_size_different_topologies():
    # The old key (n_u, COO length) aliases these two matrices. D0 -> D2
    # and dual -> single grids produced the same collision in real runs.
    r=np.array([0,1,1,2]); c=np.array([0,0,1,2]); data=np.arange(1.,5.)
    for rows,cols in [(r,c),(c,r),(r,c)]:
        got=_csc_cached(rows,cols,data,3)
        expected=sparse.csc_matrix((data,(rows,cols)),shape=(3,3))
        np.testing.assert_array_equal(got.toarray(),expected.toarray())


@pytest.mark.parametrize('multiplier',[.25,1.,4.])
def test_dpdp_tmf_jacobian_matches_all_columns(multiplier):
    grid=CartesianGrid.uniform((.2,.1,.1),.1)
    spec=fluid_from_name('example',temperature_k=350.)
    rock=DualRock.from_cf(2,k_matrix_m2=1e-15,phi_matrix=.08,cf_m2=1e-12,phi_fracture=.02)
    # Both directions of matrix-fracture transfer, with unlike compositions.
    pf=np.array([1.15e7,1.3e7]);pm=np.array([1.3e7,1.1e7])
    nf=moles_from_z(spec,pf,np.array([.55,.45]),rock.fracture.porosity*grid.cell_volumes())
    nm=moles_from_z(spec,pm,np.array([.65,.35]),rock.matrix.porosity*grid.cell_volumes())
    state=DualCompositionalState(CompositionalContinuumState(pf,nf),CompositionalContinuumState(pm,nm),0.)
    transfer=ComponentTransfer(shape_factor=40.,k_matrix_m2=1e-15,transfer_multiplier=multiplier)
    ctx=DPDPModelContext.build(grid,spec.nc);tf,tm=ctx.transmissibilities(rock)
    props_f=flash_state(spec,pf,nf);props_m=flash_state(spec,pm,nm)
    nscale=float(np.mean(nf.sum(axis=1)));pscale=float(pf.mean())
    jac,_=assemble_block_jacobian(grid,spec,rock,state,2.,transfer,tf,tm,props_f,props_m,nscale,pscale)
    def residual(u):
        n_f,p_f,n_m,p_m=unpack_dual(u,2,spec.nc)
        st=DualCompositionalState(CompositionalContinuumState(p_f,n_f),CompositionalContinuumState(p_m,n_m),0.)
        return dual_residual(grid,rock,spec,st,state,dt=2.,transfer=transfer)[0]
    u=pack_dual(state);ref=[]
    for col in range(u.size):
        eps=.1 if col%3==2 else 1e-8
        v=u.copy();v[col]+=eps
        ref.append((residual(v)-residual(u))/eps)
    fd=np.column_stack(ref)
    errors=np.linalg.norm(jac.toarray()-fd,axis=0)/np.maximum(np.linalg.norm(fd,axis=0),1e-20)
    print('DPDP Tmf multiplier',multiplier,'max column FD error',float(errors.max()))
    assert errors.max()<5e-4

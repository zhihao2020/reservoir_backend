"""Reproduce physical_3d well diagnostics without running GEM."""
from pathlib import Path
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reservoir_backend.twin.cmg_benchmark import load_twin_case, theta_true_from_twin, load_hidden_truth, rmse
from reservoir_backend.comp.properties import flash_state
from reservoir_backend.comp.wells import well_molar_sources, _wi, _injectate_xi_lam, perforation_pressures
from reservoir_backend.twin.cmg_benchmark import parse_gem_well_connections
from scripts.lab_v1_cmg_compare_plot import _field_at


def diagnostics(cache=None):
    case = ROOT / 'examples/lab_v1/cmg_gem/physical_3d/case.yaml'
    twin = load_twin_case(case)
    theta = theta_true_from_twin(twin)
    rock = twin.rock_from_theta(theta)
    times = np.array([0., 8.64, 864.])
    if cache is None:
        traj = twin.simulate(parameters=theta, t_end=864., report_times=times)
    else:
        # Composition and pressure suffice for source diagnostics; never infer
        # the total in-place moles from this pressure/overall-z comparison cache.
        from types import SimpleNamespace
        with np.load(cache) as saved:
            times = saved['times_s'].copy()
            if bool(saved['truncated']) or times[-1] != 864.:
                raise ValueError('diagnostics need an untruncated 864 s cache')
            states = [SimpleNamespace(pressure=p.copy(),moles=z.copy())
                      for p,z in zip(saved['pressure'],saved['z'])]
            traj = SimpleNamespace(times_s=times,states=states,reports=[])
    np.testing.assert_allclose(traj.times_s, times, rtol=0, atol=1e-9)
    truth = load_hidden_truth(case.parent / 'export/hidden')
    source = ROOT / 'results/lab_v1/cmg_gem_physical_3d/sanwei_co2.out'
    raw = source.read_text(errors='replace')
    connections = parse_gem_well_connections(raw, 864.)
    # Parse printed reservoir totals, retaining the report's sign and units.
    block = raw.split('Well Summary at Reservoir Conditions at 1.0000E-02 days', 1)[1].split('Cumulative Field Total', 1)[0]
    gem = {}
    name = None
    for line in block.splitlines():
        match = re.match(r'\s*\d+\s+(INJ|PROD\d+)\s+BHP\s+\S+\s+(\S+)', line)
        if match:
            name = match[1]
            gem[name] = {'bhp_reference_pa': float(match[2])*1000}
        if line.strip().startswith('Total'):
            vals = [float(v) for v in line.split()[1:]]
            gem[name]['reservoir_total_m3_day'] = vals[3]
    controls = {(c.port_name,c.kind):c for c in twin.experiment.controls}
    result = {'source': str(source.relative_to(ROOT)), 'geometry_echo': sorted(set(re.findall(r'\*GEOMETRY[^\n]+',raw))),
              'gem_864_wells': gem, 'gem_864_connections': connections, 'gem_8_64_wells': None,
              'note': 'No GEM well report at 8.64 s; maps only are interpolated. GEM WI is not printed here; ours is a geometric calculation, not a measured GEM WI.', 'snapshots': []}
    for t,st in zip(times[1:],traj.states[1:]):
        props = flash_state(twin.physics.fluid,st.pressure,st.moles)
        pg = _field_at(truth.times_s,truth.pressure,t)
        entry = {'t_s':float(t),'rmse_p_pa':rmse(st.pressure,pg),'gem_map_min_max_pa':[float(pg.min()),float(pg.max())], 'ours_min_max_pa':[float(st.pressure.min()),float(st.pressure.max())], 'wells':[]}
        for port in twin.ports:
            cells=port.cell_ids
            wi=np.array([_wi(twin.grid,rock,port,int(c)) for c in cells])
            q,r,b=well_molar_sources(twin.grid,rock,[port],controls,st.pressure,props,twin.physics.fluid,t)
            pw=perforation_pressures(twin.grid,port,twin.physics.fluid,props,b[port.name])
            dp=pw-st.pressure[cells]
            connection_table=[]
            for c, pwc, dpc in zip(cells,pw,dp):
                i,j,k=twin.grid.ijk(int(c))
                ijk=[i+1,j+1,twin.grid.nz-k]
                gemrow=next(row for row in connections if row['well']==port.name and row['ijk']==ijk)
                connection_table.append({'gem_ijk':ijk,'ours_bhp_pa':float(pwc),'ours_bhp_minus_block_pa':float(dpc),'gem_bhp_minus_block_pa':gemrow['bhp_minus_block_pa']})
            lam=props.lam_l[cells]+props.lam_v[cells]+props.lam_w[cells]
            if port.role=='injector':
                _,li=_injectate_xi_lam(twin.physics.fluid,b[port.name]);lam=np.where(dp>=0,li,lam)
            if port.use_productivity and not port.allow_crossflow:
                dp=np.maximum(dp,0.) if port.role=='injector' else np.minimum(dp,0.)
            entry['wells'].append({'well':port.name,'connection_table':connection_table,'connections':len(cells),'bhp_pa':b[port.name], 'ours_block_min_max_pa':[float(st.pressure[cells].min()),float(st.pressure[cells].max())], 'gem_map_block_min_max_pa':[float(pg[cells].min()),float(pg[cells].max())], 'wi_per_connection_m3':wi.tolist(), 'wi_total_m3':float(wi.sum()), 'reservoir_m3_day':float(np.sum(wi*lam*dp)*86400), 'source_mol_s':float(q.sum()), 'component_source_mol_s':q.sum(axis=0).tolist()})
        result['snapshots'].append(entry)
    result['cache']=None if cache is None else str(cache)
    result['accepted_steps']=len(traj.reports) if traj.reports else None
    result['max_mass_relative_error']=max((r.mass.relative_balance_error for r in traj.reports),default=None)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache',type=Path)
    args=parser.parse_args()
    result=diagnostics(args.cache)
    dest=ROOT/'results/lab_v1/cmg_gem_physical_3d_compare/well_diagnostics.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

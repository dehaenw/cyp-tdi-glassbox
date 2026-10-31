"""Draw an explanation. Every Morgan term is a real substructure, so it can be
highlighted on the molecule that triggered it -- the coefficient and the picture
are the same object, which is the whole argument for keeping the fingerprint
unfolded."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

POS, NEG = (0.96, 0.52, 0.45), (0.45, 0.68, 0.95)   # toward TDI / away from TDI


def _env_atoms(mol, bit, radius):
    """Atoms of every occurrence of one Morgan environment in this molecule."""
    ao = Chem.rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2).GetSparseCountFingerprint(
        mol, additionalOutput=ao)
    hits = []
    for at, rad in ao.GetBitInfoMap().get(bit, ()):
        if radius >= 0 and rad != radius:
            continue
        if rad == 0:
            hits.append([at]); continue
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at)
        hits.append(sorted({a for b in env for a in
                            (mol.GetBondWithIdx(b).GetBeginAtomIdx(),
                             mol.GetBondWithIdx(b).GetEndAtomIdx())}))
    return hits


def highlight_terms(smiles, explain_df, top=6, size=(300, 260)):
    """-> (list of SVG strings, the rows drawn). One panel per Morgan term,
    coloured by the SIGN of its contribution to this molecule's log-odds."""
    mol = Chem.MolFromSmiles(smiles)
    rows = explain_df[explain_df.block == 'morgan'].head(top)
    if 'bit' not in rows:
        raise ValueError('explain_df must carry the bit/radius columns '
                         '(produced by GlassBox.explain)')
    svgs = []
    for _, r in rows.iterrows():
        atoms = [a for h in _env_atoms(mol, int(r.bit), int(r.radius)) for a in h]
        col = POS if r.contribution > 0 else NEG
        d = rdMolDraw2D.MolDraw2DSVG(*size)
        d.drawOptions().addStereoAnnotation = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=sorted(set(atoms)),
            highlightAtomColors={a: col for a in set(atoms)},
            legend=f'{r.term}  {r.contribution:+.2f}')
        d.FinishDrawing(); svgs.append(d.GetDrawingText())
    return svgs, rows


def waterfall(explain_df, top=14, width=760, row_h=22):
    """Inline SVG of the biggest contributions, in log-odds. The bars sum to
    logit - intercept exactly; nothing is normalised or rescaled."""
    d = explain_df.head(top)
    a = explain_df.attrs
    m = float(np.abs(d.contribution).max()) or 1.0
    x0, lab = width * 0.52, width * 0.50
    h = row_h * len(d) + 54
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" '
           f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="12">',
           f'<text x="8" y="16" font-size="13" font-weight="600">'
           f'{a["isoform"]}  p(TDI)={a["probability"]:.3f}  '
           f'(intercept alone {a["baseline_probability"]:.3f})</text>',
           f'<line x1="{x0}" y1="26" x2="{x0}" y2="{h-20}" stroke="#999"/>']
    for i, (_, r) in enumerate(d.iterrows()):
        y = 38 + i * row_h
        w = abs(r.contribution) / m * (width * 0.44)
        x = x0 if r.contribution > 0 else x0 - w
        c = '#e2574c' if r.contribution > 0 else '#5f8fd4'
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h-7}" fill="{c}"/>')
        out.append(f'<text x="{lab:.0f}" y="{y+row_h-11}" text-anchor="end" fill="#333">'
                   f'{r.readable[:58]}</text>')
        out.append(f'<text x="{x0 + (w if r.contribution>0 else -w) + (5 if r.contribution>0 else -5):.0f}"'
                   f' y="{y+row_h-11}" text-anchor="{"start" if r.contribution>0 else "end"}"'
                   f' fill="#555">{r.contribution:+.2f}</text>')
    out.append(f'<text x="8" y="{h-6}" fill="#666">bars are log-odds; '
               f'red pushes toward TDI, blue away</text></svg>')
    return '\n'.join(out)


def grid(svgs, per_row=3):
    cells = ''.join(f'<div>{s}</div>' for s in svgs)
    return (f'<div style="display:grid;grid-template-columns:repeat({per_row},1fr);'
            f'gap:4px">{cells}</div>')

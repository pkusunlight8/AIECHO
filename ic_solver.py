"""Original ECHO-IC pure solver functions; workspace orchestration omitted."""
import numpy as np
from scipy.special import logsumexp
FAMILIES = ['GPT','Gemini','Claude','DeepSeek','Kimi','GLM','Xiaomi MiMo','Qwen','Grok']
PATTERNS = np.array([[(mask >> j) & 1 for j in range(9)] for mask in range(512)], float)
A = PATTERNS.T
PRIOR_FLOOR = 1e-15

def positive_prior(q):
    q=np.maximum(np.asarray(q,float),PRIOR_FLOOR)
    return q/q.sum()

def state(theta,logq,r,d):
    logits=logq-A.T@theta;logz=logsumexp(logits);p=np.exp(logits-logz);m=A@p
    cov=(A*p)@A.T-np.outer(m,m);cov=(cov+cov.T)/2
    value=float(logz+theta@r+.5*(d*theta)@theta)
    grad=r-m+d*theta;hessian=cov+np.diag(d)
    return value,grad,hessian,p,m,cov,float(logz)

def solve(q,r,variance,gamma,start=None):
    """Minimize the exact strictly convex nine-variable dual by damped Newton."""
    q=positive_prior(q);r=np.asarray(r,float);variance=np.asarray(variance,float)
    assert np.isfinite(r).all() and np.all(variance>0) and gamma>0
    logq=np.log(q);d=variance/gamma;theta=np.zeros(9) if start is None else np.asarray(start,float).copy()
    line_search_steps=0;converged=False
    for iteration in range(1,151):
        value,grad,H,p,m,cov,logz=state(theta,logq,r,d)
        moment_relation=theta-(m-r)/d
        normalized_kkt=float(np.max(np.abs(moment_relation)/(1+np.abs(theta))))
        if normalized_kkt<2e-8 and np.max(np.abs(grad))<1e-11:
            converged=True;break
        step=np.linalg.solve(H,grad);descent=float(grad@step)
        arithmetic_tolerance=32*np.finfo(float).eps*(1+abs(logz)+np.abs(theta*r).sum()+.5*(d*theta**2).sum())
        step_scale=1.;accepted=False
        for backtrack in range(60):
            candidate=theta-step_scale*step
            trial=state(candidate,logq,r,d)
            # A full Newton step can improve the derivative after objective
            # differences have fallen below double-precision resolution.
            objective_ok=trial[0]<=value-1e-4*step_scale*descent+2e-16*max(1.,abs(value))
            derivative_ok=np.linalg.norm(trial[1]/np.sqrt(np.diag(trial[2])))<=np.linalg.norm(grad/np.sqrt(np.diag(H)))
            precision_refinement=descent<arithmetic_tolerance and derivative_ok and trial[0]<=value+arithmetic_tolerance
            if (objective_ok and descent>=arithmetic_tolerance) or precision_refinement:
                theta=candidate;accepted=True;line_search_steps+=backtrack;break
            step_scale*=.5
        if not accepted:
            raise RuntimeError(f'Newton line search failed: grad={np.max(np.abs(grad))}, KKT={normalized_kkt}')
    value,grad,H,p,m,cov,logz=state(theta,logq,r,d)
    relation=theta-(m-r)/d
    norm_kkt=float(np.max(np.abs(relation)/(1+np.abs(theta))))
    if not converged and norm_kkt>=2e-7:
        raise RuntimeError(f'Dual did not converge: grad={np.max(np.abs(grad))}, KKT={norm_kkt}, theta={theta.tolist()}, r={r.tolist()}, variance={variance.tolist()}, gamma={gamma}')
    logp=logq-A.T@theta-logz
    kl=float(p@(logp-np.log(q)))
    fit=float(.5*gamma*np.sum((m-r)**2/variance))
    primal=kl+fit;dual=-value
    primal_gradient=logp-np.log(q)+(A.T@((m-r)/d))
    stationary=float(np.max(np.abs(primal_gradient-(p@primal_gradient))))
    eig=np.linalg.eigvalsh(H)
    inverse=np.linalg.solve(H,np.eye(9));B=p[:,None]*(A.T-m)
    Jp=B@inverse;Jm=cov@inverse
    # Jm is similar to Cw(Cw+I)^-1; the whitened spectrum lies in [0,1).
    Cw=cov/np.sqrt(d[:,None]*d[None,:]);cw_eig=np.linalg.eigvalsh(Cw)
    spectrum=cw_eig/(1+cw_eig)
    diagnostic=dict(iterations=iteration,line_search_steps=line_search_steps,dual_gradient_inf=float(np.max(np.abs(grad))),
      normalized_kkt_residual=norm_kkt,primal_stationarity_inf=stationary,normalization_residual=abs(p.sum()-1.),
      marginal_identity_residual=float(np.max(np.abs(A@p-m))),primal_objective=primal,dual_objective=dual,
      primal_dual_gap=primal-dual,kl_from_prior=kl,uncertainty_weighted_fit=fit,
      hessian_min_eigenvalue=float(eig.min()),hessian_condition=float(eig.max()/eig.min()),
      whitened_sensitivity_min_eigenvalue=float(spectrum.min()),whitened_sensitivity_max_eigenvalue=float(spectrum.max()),
      sensitivity_mass_residual=float(np.max(np.abs(Jp.sum(axis=0)))),minimum_pattern_probability=float(p.min()),
      maximum_pattern_probability=float(p.max()),empty_probability=float(p[0]),any_family_probability=float(1-p[0]),
      entropy_nats=float(-p@logp),maximum_absolute_marginal_adjustment=float(np.max(np.abs(m-r))),
      floating_underflow_atoms=int((p==0).sum()),minimum_log_probability=float(logp.min()))
    assert p.min()>=0 and p.max()<=1 and np.isfinite(logp).all() and diagnostic['normalization_residual']<2e-12
    assert diagnostic['primal_dual_gap']>=-2e-10 and abs(diagnostic['primal_dual_gap'])<2e-8
    assert eig.min()>0 and diagnostic['normalized_kkt_residual']<2e-7
    assert spectrum.min()>-1e-7 and spectrum.max()<1+1e-7
    assert np.all(m>=0) and np.all(m<=1)
    assert m.max()<=1-p[0]+1e-12 and 1-p[0]<=m.sum()+1e-12
    return dict(p=p,logp=logp,m=m,cov=cov,Jp=Jp,Jm=Jm,theta=theta,diagnostic=diagnostic)

def make_psd(matrix):
    matrix=(matrix+matrix.T)/2;eig,V=np.linalg.eigh(matrix)
    repaired=(V*np.maximum(eig,0.))@V.T
    return repaired,float(eig.min()),float(np.linalg.norm(repaired-matrix,'fro'))

"""Evaluation-only interhand geometry capture. Never imported by a policy."""
import numpy as np
import mujoco

class NativeInterhandGeometryAudit:
    def __init__(self,model):
        self.model=model;sides={}
        for side in ('left','right'):
            bodies=[i for i in range(model.nbody)if (model.body(i).name or'').startswith('aero_'+side+'_')]
            sides[side]=np.array([i for i in range(model.ngeom)if model.geom_bodyid[i]in bodies and(model.geom_contype[i]or model.geom_conaffinity[i])])
        left,right=np.meshgrid(sides['left'],sides['right'],indexing='ij');self.left=left.ravel();self.right=right.ravel();self.radii=model.geom_rbound[self.left]+model.geom_rbound[self.right];self.rows=[]
    def record(self,data):
        lower=np.linalg.norm(data.geom_xpos[self.left]-data.geom_xpos[self.right],axis=1)-self.radii
        closest=(.02,-1,-1)
        for i in np.flatnonzero(lower<.02):
            a,b=int(self.left[i]),int(self.right[i]);distance=float(mujoco.mj_geomDistance(self.model,data,a,b,.03,None))
            if distance<closest[0]:closest=(distance,a,b)
        self.rows.append((float(data.time),*closest))
    def arrays(self):
        return {'interhand_geometry':np.asarray(self.rows,dtype=float).reshape(-1,4),'interhand_geometry_columns':np.asarray(['timestamp_s','minimum_distance_m','left_geom_id','right_geom_id'])}
    def summary(self):
        if not self.rows:return dict(evaluation_only=True,available=False,sampling_hz=1./self.model.opt.timestep,minimum_interhand_geometry_m=None,exceeds_predeclared_5mm_penetration=None,negative_samples=0)
        rows=np.asarray(self.rows);i=int(np.argmin(rows[:,1]));row=rows[i];return dict(evaluation_only=True,sampling_hz=1./self.model.opt.timestep,minimum_interhand_geometry_m=float(row[1]),minimum_time_s=float(row[0]),left_geom=None if row[2]<0 else self.model.geom(int(row[2])).name,right_geom=None if row[3]<0 else self.model.geom(int(row[3])).name,exceeds_predeclared_5mm_penetration=bool(row[1]<-.005),negative_samples=int((rows[:,1]<0).sum()),geometry_probe_is_read_only=True)

class NativeBallContactWitness:
    """Full per-contact geometry/force witness, evaluation-only after each step."""
    columns=('timestamp_s','other_geom_id','other_body_id','distance_m','normal_n','vertical_n','contact_x_m','contact_y_m','contact_z_m','normal_on_ball_x','normal_on_ball_y','normal_on_ball_z')
    def __init__(self,model):
        self.model=model;self.ball=int(model.geom('football_surface').id);self.rows=[];self.force=np.zeros(6)
    def record(self,data):
        for i,c in enumerate(data.contact):
            a,b=map(int,c.geom)
            if a==self.ball:other,sign=b,-1.
            elif b==self.ball:other,sign=a,1.
            else:continue
            mujoco.mj_contactForce(self.model,data,i,self.force)
            if self.force[0]<=0. and c.dist>=0.:continue
            frame=np.asarray(c.frame).reshape(3,3);normal=sign*frame[0];vertical=sign*float(frame[:,2]@self.force[:3])
            self.rows.append((float(data.time),other,int(self.model.geom_bodyid[other]),float(c.dist),max(0.,float(self.force[0])),vertical,*c.pos,*normal))
    def arrays(self):
        return {'ball_contact_witness':np.asarray(self.rows,float).reshape(-1,len(self.columns)),'ball_contact_witness_columns':np.asarray(self.columns),'geom_names':np.asarray([self.model.geom(i).name or''for i in range(self.model.ngeom)]),'body_names':np.asarray([self.model.body(i).name or''for i in range(self.model.nbody)])}

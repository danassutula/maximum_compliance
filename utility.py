import dolfin

mesh = dolfin.UnitSquareMesh(10,10)
V = dolfin.FucntionSpace(mesh, 'CG', 1)
u = dolfin.Function(V)

dolfin.File('saved_mesh.xml') << mesh
dolfin.File('saved_u_CG_1.xml') << u

mesh_old = dolfin.Mesh('saved_mesh.xml')
V_new = dolfin.FucntionSpace(mesh_old, 'CG', 1 )
u_old = dolfin.Function(V_old, 'saved_u_CG_1.xml')

mesh_new = dolfin.UnitSquareMesh(100, 100)
V_new = dolfin.FunctionSpace(mesh_new, 'CG', 1)
u_new = dolfin.project(u_old, V_new)

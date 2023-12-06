import taichi as ti
import taichi.math as tm
import torch
import numpy as np

@ti.data_oriented
class MPM:
    def __init__(self, n_rods, n_vertices):
        # Only simulate Hair, so no (i) particles
        # parameters obtained from the paper Table1 and Table2
        self.n_grid = 1000
        self.dx = 1e-2   # grid size is 10
        self.inv_dx = 1 / self.dx
        self.rho = 1
        self.dt = 1e-4         # time step
        self.E = 60            # Young's modulus
        self.gamma = 10        # shear modulus
        self.k = 2000          # stiffness
        self.nu = 0.3          # Poisson's ratio
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.n_rods = n_rods
        self.d_num2 = n_vertices
        self.d_num3 = self.d_num2-1
        self.num2 = self.n_rods*self.d_num2
        self.num3 = self.n_rods*self.d_num3
        self.damping = 0.07       # damping inner motion
        self.alpha = 0.154        # anisotropic friction
        self.beta = 0.268

        # (ii) particles info
        self.x2 = ti.Vector.field(3, dtype=float, shape=(self.n_rods, self.d_num2))
        self.v2 = ti.Vector.field(3, dtype=float, shape=(self.n_rods, self.d_num2))
        self.C2 = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num2))
        self.volumn2 = ti.field(dtype=float, shape=(self.n_rods, self.d_num2))

        # (iii) particles info
        self.x3 = ti.Vector.field(3, dtype=float, shape=(self.n_rods, self.d_num3))
        self.v3 = ti.Vector.field(3, dtype=float, shape=(self.n_rods, self.d_num3))
        self.C3 = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num3))
        self.F3 = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num3))  # elastic deformation
        self.Q3 = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num3))  # QR decomposition
        self.R3 = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num3))  # QR decomposition
        self.D3_inv = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num3))
        self.d3 = ti.Matrix.field(3, 3, dtype=float, shape=(self.n_rods, self.d_num3))  # tangent and normal
        self.volumn3 = ti.field(dtype=float, shape=(self.n_rods, self.d_num3))

        # grid info
        self.grid_mv = ti.Vector.field(3, dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))  # momentum
        self.grid_m = ti.field(dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_v = ti.Vector.field(3, dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_f = ti.Vector.field(3, dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_tag = ti.field(dtype=int, shape=(self.n_grid, self.n_grid, self.n_grid))       # if m = 0, tag = 0

        # auto diff: torch  ||  to ti.Matrix: numpy
        self.R3_t = torch.empty((self.n_rods, self.d_num3, 3, 3), dtype=torch.float32)
        self.Q3_t = torch.empty((self.n_rods, self.d_num3, 3, 3), dtype=torch.float32)
        self.R3_n_grad = np.empty((self.n_rods, self.d_num3, 3, 3), dtype=np.float32)
        self.Q3_n_grad = np.empty((self.n_rods, self.d_num3, 3, 3), dtype=np.float32)

    
    @ti.func
    def QR3(self, A: ti.types.template(), Q: ti.types.template(), R: ti.types.template()):
        # QR decomposition
        for i in range(A.shape[1]):
            v = A[:, i]
            for j in range(i):
                R[j, i] = Q[:, j].dot(A[:, i])
                v = v - R[j, i] * Q[:, j]
            R[i, i] = tm.length(v)
            Q[:, i] = v / R[i, i]

    @ti.func
    def LDU(self, A: ti.types.template(), L: ti.types.template(), D: ti.types.template(), U: ti.types.template()):
        # LDU decomposition
        for i in range(A.shape[1]):
            for j in range(i):
                L[i, j] = A[i, j]
                for k in range(j):
                    L[i, j] -= L[i, k] * D[k, k] * U[k, j]
                L[i, j] /= D[j, j]
            for j in range(i, A.shape[1]):
                U[i, j] = A[i, j]
                for k in range(i):
                    U[i, j] -= L[i, k] * D[k, k] * U[k, j]
            D[i, i] = A[i, i]
            for k in range(i):
                D[i, i] -= L[i, k] * D[k, k] * U[k, i]
    
    # paper and supplement are contradictory, paper：R3 decomposition, supplement：R3 slice, which is simple
    def Compute_energy_grad(self, i, j):
        r11 = self.R3_t[0, 0]
        r12 = self.R3_t[0, 1]
        r13 = self.R3_t[0, 2]
        f = 0.5 * 2000 * (r11 - 1) ** 2
        g = 0.5 * 10 * (r12 ** 2 + r13 ** 2)
        U, S, V = torch.svd(self.R3_t[1:, 1:])
        s = torch.log(S)
        h = 23.077 * (s[0] ** 2 + s[1] ** 2) + 0.5 * 34.615 * (s[0] + s[1]) ** 2
        energy = f + g + h
        energy.backward()
        self.R3_n_grad[i, j] = self.R3_t.grad.numpy()[i, j]


    # Compute stress
    @ti.kernel
    def Compute_Piola_Kirchhoff(self, i, j)-> ti.types.template():
        self.Compute_energy_grad(i, j)
        de_dR = ti.Matrix(self.R3_n_grad[i, j])
        K = de_dR @ (self.R3[i, j].transpose())
        L = ti.Matrix.zero(float, 3, 3)
        D = ti.Matrix.zero(float, 3, 3)
        U = ti.Matrix.zero(float, 3, 3)
        self.LDU(K, L, D, U)
        de_dd = self.Q3[i, j] @ (U + U.transpose() - D) @ ti.Matrix.inverse(self.R3[i, j]).transpose()
        P = de_dd @ ti.Matrix.inverse(self.D3_inv[i, j]).transpose() @ ti.Matrix.inverse(self.F3[i, j]).transpose()
        stress = 1/tm.determinant(self.F3[i, j]) * P @ self.F3[i, j].transpose()
        return stress

    @ti.kernel
    def Particle_to_Gird(self):
        for i, j, k in self.grid_m:
            self.grid_m[i, j, k] = 0
            self.grid_mv[i, j, k] = ti.Vector.zero(float, 3)

        for i, j in self.x2:
            base = (self.x2[i, j] * self.inv_dx - 0.5).cast(int)
            fx = self.x2[i, j] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            affine = self.C2[i, j]
            mass = self.volumn2[i, j] * self.rho
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                self.grid_m[base + offset] += weight * mass
                dpos = (offset.cast(float) - fx) * self.dx
                self.grid_mv[base + offset] += weight * mass * (self.v2[i, j] + affine @ dpos)

        for i, j in self.x3:
            base = (self.x3[i, j] * self.inv_dx - 0.5).cast(int)
            fx = self.x3[i, j] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            affine = self.C3[i, j]
            mass = self.volumn3[i, j] * self.rho
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                self.grid_m[base + offset] += weight * mass
                dpos = (offset.cast(float) - fx) * self.dx
                self.grid_mv[base + offset] += weight * mass * (self.v3[i, j] + affine @ dpos)

    @ti.kernel
    def Grid_velocity(self):
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                self.grid_v[i, j, k] = self.grid_mv[i, j, k] / self.grid_m[i, j, k]
                self.grid_tag[i, j, k] = 1
            else:
                self.grid_tag[i, j, k] = 0

    @ti.kernel
    def Grid_Force(self):
        for i, j, k in self.grid_f:
            self.grid_f[i, j, k] = 0
        
        # (ii) particles
        # only streching forces for test
        for i, j in self.x3:
            l = ti.Vector([self.d3[i, j][0, 0], self.d3[i, j][1, 0], self.d3[i, j][2, 0]])
            D = ti.Matrix.inverse(self.D3_inv[i, j])
            l_bar = ti.Vector([D[0, 0], D[1, 0], D[2, 0]])
            f = self.k * (tm.length(l) / tm.length(l_bar) - 1) * tm.normalize(l)
            base1 = (self.x2[i, j] * self.inv_dx - 0.5).cast(int)
            base2 = (self.x2[i, j+1] * self.inv_dx - 0.5).cast(int)
            fx1 = self.x2[i, j] * self.inv_dx - base1.cast(float)
            fx2 = self.x2[i, j+1] * self.inv_dx - base2.cast(float)
            w1 = [0.5 * (1.5 - fx1) ** 2, 0.75 - (fx1 - 1) ** 2, 0.5 * (fx1 - 0.5) ** 2]
            w2 = [0.5 * (1.5 - fx2) ** 2, 0.75 - (fx2 - 1) ** 2, 0.5 * (fx2 - 0.5) ** 2]
            for offset1 in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                if self.grid_tag[base1 + offset1] == 1:
                    weight1 = w1[offset1[0]][0] * w1[offset1[1]][1] * w1[offset1[2]][2]
                    self.grid_f[base1 + offset1] += weight1 * f
            for offset2 in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                if self.grid_tag[base2 + offset2] == 1:
                    weight2 = w2[offset2[0]][0] * w2[offset2[1]][1] * w2[offset2[2]][2]
                    self.grid_f[base2 + offset2] -= weight2 * f
            
        # (iii) particles
        for i, j in self.x3:
            self.Compute_energy_grad(i, j)
            de_dR = ti.Matrix(self.R3_n_grad[i, j])
            K = de_dR @ (self.R3[i, j].transpose())
            L = ti.Matrix.zero(float, 3, 3)
            D = ti.Matrix.zero(float, 3, 3)
            U = ti.Matrix.zero(float, 3, 3)
            self.LDU(K, L, D, U)
            de_dd = self.Q3[i, j] @ (U + U.transpose() - D) @ ti.Matrix.inverse(self.R3[i, j]).transpose()
            de_dF = de_dd @ ti.Matrix.inverse(self.D3_inv[i, j]).transpose()
            c1 = ti.Vector([de_dF[0, 1], de_dF[1, 1], de_dF[2, 1]])
            c2 = ti.Vector([de_dF[0, 2], de_dF[1, 2], de_dF[2, 2]])
            d1 = ti.Vector([self.d3[i, j][0, 1], self.d3[i, j][1, 1], self.d3[i, j][2, 1]])
            d2 = ti.Vector([self.d3[i, j][0, 2], self.d3[i, j][1, 2], self.d3[i, j][2, 2]])

            # grad_weight
            base = (self.x3[i, j] * self.inv_dx - 0.5).cast(int)
            fx = self.x3[i, j] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                if self.grid_tag[base + offset] == 1:
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                    w_grad = 4 * weight * dpos * (self.inv_dx ** 2)
                    f = - self.volumn3[i, j] * (d1.dot(w_grad) * c1 + d2.dot(w_grad) * c2)
                    self.grid_f[base + offset] += f

    @ti.kernel
    def QR_decomposition(self):
        for i, j in self.x3:
            self.QR3(self.d3[i, j], self.Q3[i, j], self.R3[i, j])

    def trans_QR(self):
        self.Q3_t = self.Q3.to_torch(keep_dims = True)
        self.R3_t = self.R3.to_torch(keep_dims = True)
        self.Q3_t.requires_grad = True
        self.R3_t.requires_grad = True

    @ti.kernel
    def Update_Grid(self):
        for i, j, k in self.grid_tag:
            if self.grid_tag[i, j, k] == 1:
                self.grid_v[i, j, k] += self.dt * self.grid_f[i, j, k] / self.grid_m[i, j, k]
                self.grid_v[i, j, k] += self.dt * ti.Vector([0, -981.0, 0])    # different unit

                # Boundary condition
                if i < 3 and self.grid_v[i, j, k][0] < 0:
                    self.grid_v[i, j, k][0] = 0
                if i > self.n_grid - 3 and self.grid_v[i, j, k][0] > 0:
                    self.grid_v[i, j, k][0] = 0
                if j < 3 and self.grid_v[i, j, k][1] < 0:
                    self.grid_v[i, j, k][1] = 0
                if j > self.n_grid - 3 and self.grid_v[i, j, k][1] > 0:
                    self.grid_v[i, j, k][1] = 0
                if k < 3 and self.grid_v[i, j, k][2] < 0:
                    self.grid_v[i, j, k][2] = 0
                if k > self.n_grid - 3 and self.grid_v[i, j, k][2] > 0:
                    self.grid_v[i, j, k][2] = 0

    @ti.kernel
    def Update_Deformation_Gradient(self):
        # Lagrangian and Eularian gradient
        for i, j in self.x3:
            dc1 = self.x2[i, j+1] - self.x2[i, j]
            dc2 = ti.Vector(self.d3[i, j][0, 1], self.d3[i, j][1, 1], self.d3[i, j][2, 1])
            dc3 = ti.Vector(self.d3[i, j][0, 2], self.d3[i, j][1, 2], self.d3[i, j][2, 2])
            dc2 += self.dt * self.C3[i, j] @ dc2
            dc3 += self.dt * self.C3[i, j] @ dc3
            self.d3[i, j] = ti.Matrix.cols([dc1, dc2, dc3])
            self.F3[i, j] = self.d3[i, j] @ self.D3_inv[i, j]
            # d3 and F3 need to go through Return Mapping

    @ti.kernel
    def Grid_to_Particle(self):
        # Only update positions of (i) and (ii),
        # and calculate (iii) as the barycenters of (ii)
        for i, j in self.x2:
            base = (self.x2[i, j] * self.inv_dx - 0.5).cast(int)
            fx = self.x2[i, j] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                if self.grid_tag[base + offset] == 1:
                    dpos = (offset.cast(float) - fx) * self.dx
                    g_v = self.grid_v[base + offset]
                    weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * (self.inv_dx ** 2)
            
            # Damping Cp
            self.v2[i, j] = new_v
            C_s = 0.5 * (new_C + new_C.transpose())
            C_k = new_C - C_s
            self.C2[i, j] = C_k + self.damping * C_s
        
        for i, j in self.x3:
            base = (self.x3[i, j] * self.inv_dx - 0.5).cast(int)
            fx = self.x3[i, j] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_C = ti.Matrix.zero(float, 3, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                if self.grid_tag[base + offset] == 1:
                    dpos = (offset.cast(float) - fx) * self.dx
                    g_v = self.grid_v[base + offset]
                    weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                    new_C += 4 * weight * g_v.outer_product(dpos) * (self.inv_dx ** 2)

            # Damping Cp
            C_s = 0.5 * (new_C + new_C.transpose())
            C_k = new_C - C_s
            self.C3[i, j] = C_k + self.damping * C_s

    @ti.kernel
    def Update_Particle(self):
        for i, j in self.x2:
            self.x2[i, j] += self.dt * self.v2[i, j]
        
        for i, j in self.x3:
            self.x3[i, j] = 0.5 (self.x2[i, j] + self.x2[i, j+1])
            self.v3[i, j] = 0.5 (self.v2[i, j] + self.v2[i, j+1])

    @ti.kernel
    def Return_Mapping(self):
        # QR decomposition, change R
        for i, j in self.x3:
            sR3 = self.R3[i, j]
            R3 = ti.Matrix([[sR3[1, 1], sR3[1, 2]], [sR3[2, 1], sR3[2, 2]]])
            U, S, V = ti.svd(R3)
            E = tm.log(S)
            e1 = E[0, 0]
            e2 = E[1, 1]
            if e1 < e2:
                e1, e2 = e2, e1
            stress = self.Compute_Piola_Kirchhoff(i, j)
            J2 = (stress[1, 1] - stress[2, 2]) ** 2 + 4 * stress[1, 2]
            condition1 = tm.sqrt(J2) + 0.5 * self.alpha * (stress[1, 1] + stress[2, 2])
            condition2 = tm.sqrt(stress[0, 1] ** 2 + stress[0, 2] ** 2) + 0.5 * self.beta * (stress[1, 1] + stress[2, 2])

            # volume preserving (R3)
            if e1 + e2 >= 0:
                e1 = e2 = 0
            elif condition1 > 0:
                n = 0.5 * (e1 - e2) + ((self.alpha * self.lam) / (4 * self.mu)) * (e1 + e2)
                e1 -= n
                e2 -= n
            E[0, 0] = e1
            E[1, 1] = e2
            R3 = U @ tm.exp(E) @ V.transpose()
            
            # sheering tangent (R2)
            if condition2 > 0:
                scale = - (self.beta * (stress[1, 1] + stress[2, 2])) / (2 * ti.sqrt(stress[0, 1] ** 2 + stress[1, 2] ** 2))
                self.R3[i, j][0, 1] *= scale
                self.R3[i, j][0, 2] *= scale
            
            self.R3[i, j][1, 1] = R3[0, 0]
            self.R3[i, j][1, 2] = R3[0, 1]
            self.R3[i, j][2, 1] = R3[1, 0]
            self.R3[i, j][2, 2] = R3[1, 1]
            self.d3[i, j] = self.Q3[i, j] @ self.R3[i, j]
            self.F3[i, j] = self.d3[i, j] @ self.D3_inv[i, j]
    
    def grad_zero(self):
        self.R3_t.grad.zero_()
        self.Q3_t.grad.zero_()
            
    def Step(self):
        # standard MPM pipeline
        self.Particle_to_Gird()
        self.Grid_velocity()
        
        # ti.Matrix.field to tensor, to_tensor is a kernel
        self.QR_decomposition()
        self.trans_QR()

        self.Grid_Force()

        self.grad_zero()

        self.Update_Grid()
        self.Grid_to_Particle()
        self.Update_Deformation_Gradient()
        self.Update_Particle()
        self.QR_decomposition()
        self.trans_QR()
        self.Return_Mapping()
        self.grad_zero()

    def initialize(self, X):
        for i in range(self.n_rods):
            for j in range(self.d_num2):
                self.x2[i, j] = X[i * self.d_num2 + j]
                self.v2[i, j] = ti.Vector.zero(float, 3)
                self.C2[i, j] = ti.Matrix.zero(float, 3, 3)
                self.volumn2[i, j] = 0
        
        for i in range(self.n_rods):
            for j in range(self.d_num3):
                self.x3[i, j] = 0.5 * (self.x2[i, j] + self.x2[i, j+1])
                self.v3[i, j] = ti.Vector.zero(float, 3)
                self.C3[i, j] = ti.Matrix.zero(float, 3, 3)
                self.F3[i, j] = ti.Matrix.identity(float, 3)
                d1 = self.x2[i, j+1] - self.x2[i, j]
                v = tm.cross(d1, ti.Vector([0, 1.0, 0]))
                if tm.length(v) < 1e-6:
                    v = tm.cross(d1, ti.Vector([1.0, 0, 0]))
                d2 = v
                d3 = tm.cross(d1, d2)
                self.d3[i, j] = ti.Matrix.cols([d1, d2, d3])
                self.D3_inv[i, j] = ti.Matrix.inverse(self.d3[i, j])
                l = self.x2[i, j+1] - self.x2[i, j]
                self.volumn3[i, j] = 0.5 * tm.length(l) * self.rho
                self.volumn2[i, j] += 0.5 * self.volumn3[i, j]
                self.volumn2[i, j+1] += 0.5 * self.volumn3[i, j]


#include <array>

using mat4 = std::array<std::array<float, 4>, 4>;

inline mat4 mm(const mat4& a, const mat4& b) {
	mat4 rst;
	memset(&rst, 0, sizeof(mat4));
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = 0; k < 4; k++)
				rst[i][k] += a[i][j] * b[j][k];
	return rst;
}

inline void mm_(mat4& dst, const mat4& a, const mat4& b) {
	memset(&dst, 0, sizeof(mat4));
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = 0; k < 4; k++)
				dst[i][k] += a[i][j] * b[j][k];
}

inline mat4 mmt(const mat4& a, const mat4& b) {
	mat4 rst;
	memset(&rst, 0, sizeof(mat4));
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = 0; k < 4; k++)
				rst[k][i] += a[i][j] * b[j][k];
	return rst;
}

inline mat4 transpose(const mat4& m) {
	mat4 rst;
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			rst[i][j] = m[j][i];
	return rst;
}

inline void addmm_(mat4& dst, const mat4& a, const mat4& b) {
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = 0; k < 4; k++)
				dst[i][k] += a[i][j] * b[j][k];
}

extern "C" __declspec(dllexport) void eval_matrix_world(
    size_t n_bones, 
    int64_t* bone_parents_arr, 
    mat4* bone_matrix_arr, 
    mat4* matrix_basis_arr, 
    mat4* matrix_world_arr
) 
{
	for (size_t i = 0; i < n_bones; i++) {
		if (bone_parents_arr[i] < 0)
			matrix_world_arr[i] = mm(bone_matrix_arr[i], matrix_basis_arr[i]);
		else
			matrix_world_arr[i] = mm(matrix_world_arr[bone_parents_arr[i]], mm(bone_matrix_arr[i], matrix_basis_arr[i]));
	}
}

extern "C" __declspec(dllexport) void grad_matrix_world(
    size_t n_bones, 
	
    int64_t* bone_parents_arr, 
    mat4* bone_matrix_arr, 
    mat4* matrix_basis_arr, 
    mat4* matrix_world_arr, 

    mat4* grad_matrix_basis_arr,
    mat4* grad_matrix_world_arr 
) 
{
    for (ptrdiff_t i = n_bones - 1; i >= 0; i--) {
        if (bone_parents_arr[i] < 0) {
            // grad to matrix_basis
            grad_matrix_basis_arr[i] = mm(transpose(bone_matrix_arr[i]), grad_matrix_world_arr[i]);
        }
        else {
            // grad to matrix_basis
            grad_matrix_basis_arr[i] = mm(mmt(matrix_world_arr[bone_parents_arr[i]], bone_matrix_arr[i]), grad_matrix_world_arr[i]);
            // grad to parent matrix_world
            addmm_(grad_matrix_world_arr[bone_parents_arr[i]], grad_matrix_world_arr[i], mmt(bone_matrix_arr[i], matrix_basis_arr[i]));
        }
    }
}

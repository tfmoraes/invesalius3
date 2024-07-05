#ifndef MESH_H
#define MESH_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <execution>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <limits>

template <typename T> struct Vertex_t {
  T x;
  T y;
  T z;

  friend std::ostream &operator<<(std::ostream &out,
                                  const Vertex_t<T> &vertex) {
    return out << "<" << vertex.x << ", " << vertex.y << ", " << vertex.z
               << ">";
  }
};

template <typename T> struct Normal_t {
  T x;
  T y;
  T z;

  friend std::ostream &operator<<(std::ostream &out,
                                  const Normal_t<T> &normal) {
    return out << "<" << normal.x << ", " << normal.y << ", " << normal.z
               << ">";
  }
};

template <typename T> struct Face_t {
  T num_vertices;
  T vertex_0;
  T vertex_1;
  T vertex_2;

  friend std::ostream &operator<<(std::ostream &out, const Face_t<T> &face) {
    return out << "<" << face.vertex_0 << ", " << face.vertex_1 << ", "
               << face.vertex_2 << ">";
  }
};

using Key_t = std::pair<int, int>;

// Define a custom hash function for std::pair<int, int>
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2; // Combine the hashes
  }
};

template <typename T>
float vertex_distance(const Vertex_t<T> &vi, const Vertex_t<T> &vj) {
  return (vi.x - vj.x) * (vi.x - vj.x) + (vi.y - vj.y) * (vi.y - vj.y) +
         (vi.z - vj.z) * (vi.z - vj.z);
}

template <typename VERTICES_TYPE, typename ID_TYPE> class MeshCPP {
public:
  std::span<Vertex_t<VERTICES_TYPE>> vertices;
  std::span<Normal_t<VERTICES_TYPE>> faces_normals;
  std::span<Face_t<ID_TYPE>> faces;

  std::unordered_map<int, std::vector<ID_TYPE>> map_vface;
  std::unordered_map<int, int> border_vertices;

  std::size_t number_of_points;
  std::size_t number_of_faces;

  bool _initialized;

  MeshCPP() {}

  MeshCPP(std::size_t number_of_points, std::size_t number_of_faces,
          VERTICES_TYPE *vertices, VERTICES_TYPE *faces_normals,
          ID_TYPE *faces) {
    this->number_of_points = number_of_points;
    this->number_of_faces = number_of_faces;
    this->vertices = std::span<Vertex_t<VERTICES_TYPE>>(
        reinterpret_cast<Vertex_t<VERTICES_TYPE> *>(vertices),
        number_of_points);
    this->faces_normals = std::span<Normal_t<VERTICES_TYPE>>(
        reinterpret_cast<Normal_t<VERTICES_TYPE> *>(faces_normals),
        number_of_faces);
    this->faces = std::span<Face_t<ID_TYPE>>(
        reinterpret_cast<Face_t<ID_TYPE> *>(faces), number_of_faces);
    this->populate_maps();
    this->_initialized = true;
    std::cout << "Number of available threads: "
              << std::thread::hardware_concurrency() << " "
              << std::this_thread::get_id() << std::endl;
    std::cout << "Number of points: " << this->number_of_points << ". Number of faces: " << this->number_of_faces << std::endl;
  }

  MeshCPP(MeshCPP &other) {
    this->_initialized = true;
    this->vertices = other.vertices;
  }

  void populate_maps(void) {
    std::unordered_map<Key_t, int, pair_hash> edge_nfaces;
    for (std::size_t i = 0; i < this->number_of_faces; i++) {
      this->map_vface[this->faces[i].vertex_0].push_back(i);
      this->map_vface[this->faces[i].vertex_1].push_back(i);
      this->map_vface[this->faces[i].vertex_2].push_back(i);
      Key_t key_0 =
          std::minmax(this->faces[i].vertex_0, this->faces[i].vertex_1);
      Key_t key_1 =
          std::minmax(this->faces[i].vertex_1, this->faces[i].vertex_2);
      Key_t key_2 =
          std::minmax(this->faces[i].vertex_0, this->faces[i].vertex_2);
      edge_nfaces[key_0] += 1;
      edge_nfaces[key_1] += 1;
      edge_nfaces[key_2] += 1;
    }

    for (const auto &[key, value] : edge_nfaces) {
      if (value == 1) {
        this->border_vertices[key.first] = 1;
        this->border_vertices[key.second] = 1;
      }
    }
  }

  void print_vertices(void) const {
    for (std::size_t i = 0; i < this->number_of_points; i++) {
      std::cout << this->vertices[i] << " ";
    }
    std::cout << '\n';

    for (std::size_t i = 0; i < this->number_of_points; i++) {
      std::cout << this->faces_normals[i] << " ";
    }
    std::cout << '\n';

    for (auto &value : this->faces) {
      std::cout << value << " ";
    }
    std::cout << '\n';
  }

  void copy_to(MeshCPP<VERTICES_TYPE, ID_TYPE> &other) {
    for (size_t i = 0; i < this->number_of_points; i++) {
      other.vertices[i] = this->vertices[i];
      other.faces_normals[i] = this->faces_normals[i];
    }
    for (size_t i = 0; i < this->number_of_faces; i++) {
      other.faces[i] = this->faces[i];
    }
    other.map_vface = this->map_vface;
    other.border_vertices = this->border_vertices;
  }

  const std::vector<ID_TYPE>& get_faces_by_vertex(ID_TYPE v_id) const {
    return this->map_vface.at(v_id);
  }

  const std::unordered_set<ID_TYPE> get_ring1(ID_TYPE v_id) const {
    std::unordered_set<ID_TYPE> ring1;
    for (auto f_id : this->map_vface[v_id]) {
      if (this->faces[f_id].vertex_0 != v_id) {
        ring1.insert(this->faces[f_id].vertex_0);
      }
      if (this->faces[f_id].vertex_1 != v_id) {
        ring1.insert(this->faces[f_id].vertex_1);
      }
      if (this->faces[f_id].vertex_2 != v_id) {
        ring1.insert(this->faces[f_id].vertex_2);
      }
    }
    return ring1;
  }

  bool is_border(ID_TYPE v_id) const {
    return this->border_vertices.find(v_id) != this->border_vertices.end();
  }

  const std::shared_ptr<std::vector<ID_TYPE>>
  get_near_vertices_to_v(ID_TYPE v_id, float dmax) const {
    std::vector<ID_TYPE> idfaces;
    auto near_vertices = std::make_shared<std::vector<ID_TYPE>>();
    std::deque<ID_TYPE> to_visit;
    std::unordered_map<ID_TYPE, bool> status_v;
    std::unordered_map<ID_TYPE, bool> status_f;
    Vertex_t<VERTICES_TYPE> vip;
    Vertex_t<VERTICES_TYPE> vjp;
    float distance;
    int vj;

    vip = this->vertices[v_id];
    to_visit.push_back(v_id);
    dmax = dmax * dmax;
    while (!to_visit.empty()) {
      v_id = to_visit.front();
      to_visit.pop_front();
      status_v[v_id] = true;

      for (auto f_id : this->get_faces_by_vertex(v_id)) {
        if (status_f.find(f_id) == status_f.end()) {
          status_f[f_id] = true;

          vj = this->faces[f_id].vertex_0;
          if (status_v.find(vj) == status_v.end()) {
            status_v[vj] = true;
            vjp = this->vertices[vj];
            distance = vertex_distance(vip, vjp);
            if (distance <= dmax) {
              near_vertices->push_back(vj);
              to_visit.push_back(vj);
            }
          }

          vj = this->faces[f_id].vertex_1;
          if (status_v.find(vj) == status_v.end()) {
            status_v[vj] = true;
            vjp = this->vertices[vj];
            distance = vertex_distance(vip, vjp);
            if (distance <= dmax) {
              near_vertices->push_back(vj);
              to_visit.push_back(vj);
            }
          }

          vj = this->faces[f_id].vertex_2;
          if (status_v.find(vj) == status_v.end()) {
            status_v[vj] = true;
            vjp = this->vertices[vj];
            distance = vertex_distance(vip, vjp);
            if (distance <= dmax) {
              near_vertices->push_back(vj);
              to_visit.push_back(vj);
            }
          }
        }
      }
    }
    return near_vertices;
  }

  void multiply_two_par(void) {
    auto startTime = std::chrono::system_clock::now();
    std::unordered_map<std::thread::id, int, std::hash<std::thread::id>>
        threads_used;
    std::mutex g_num_mutex;
    std::for_each(std::execution::par, this->vertices.begin(),
                  this->vertices.end(), [&](auto &item) {
                    item.x *= 2.0;
                    item.y *= 2.0;
                    item.z *= 2.0;
                    g_num_mutex.lock();
                    threads_used[std::this_thread::get_id()] += 1;
                    g_num_mutex.unlock();
                  });

    for (const auto &[key, value] : threads_used) {
      std::cout << "Thread " << key << ": " << value << std::endl;
    }

    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = endTime - startTime;
    std::cout << "1. Parallel: " << diff << std::endl;
  }

  void multiply_two(void) {
    auto startTime = std::chrono::system_clock::now();
    std::unordered_map<std::thread::id, int, std::hash<std::thread::id>>
        threads_used;
    std::mutex g_num_mutex;
    std::for_each(std::execution::seq, this->vertices.begin(),
                  this->vertices.end(), [&](auto &item) {
                    item.x *= 2.0;
                    item.y *= 2.0;
                    item.z *= 2.0;
                    g_num_mutex.lock();
                    threads_used[std::this_thread::get_id()] += 1;
                    g_num_mutex.unlock();
                  });

    auto endTime = std::chrono::system_clock::now();
    for (const auto &[key, value] : threads_used) {
      std::cout << "Thread " << key << ": " << value << std::endl;
    }
    std::chrono::duration<double> diff = endTime - startTime;
    std::cout << "2. Sequential: " << diff << std::endl;
  }

  void multiply_two_for_openmp(void) {
    auto startTime = std::chrono::system_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < this->vertices.size(); i++) {
      this->vertices[i].x *= 2.0;
      this->vertices[i].y *= 2.0;
      this->vertices[i].z *= 2.0;
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = endTime - startTime;
    std::cout << "3. OpenMP: " << diff << std::endl;
  }

  void multiply_two_for(void) {
    auto startTime = std::chrono::system_clock::now();
    for (size_t i = 0; i < this->vertices.size(); i++) {
      this->vertices[i].x *= 2.0;
      this->vertices[i].y *= 2.0;
      this->vertices[i].z *= 2.0;
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = endTime - startTime;
    std::cout << "4. No openmp: " << diff << std::endl;
  }
};

template <typename VERTICES_TYPE, typename ID_TYPE>
std::shared_ptr<std::vector<ID_TYPE>>
find_staircase_artifacts(const MeshCPP<VERTICES_TYPE, ID_TYPE> &mesh,
                         std::array<double, 3> stack_orientation, double T) {
  double of_x, of_z, of_y, min_x, max_x, min_y, max_y, min_z, max_z;
  auto output = std::make_shared<std::vector<ID_TYPE>>();

  for (size_t v_id = 0; v_id < mesh.number_of_points; v_id++) {
    max_z = std::numeric_limits<double>::min();
    min_z = std::numeric_limits<double>::max();
    max_y = std::numeric_limits<double>::min();
    min_y = std::numeric_limits<double>::max();
    max_x = std::numeric_limits<double>::min();
    min_x = std::numeric_limits<double>::max();
    for (auto f_id : mesh.get_faces_by_vertex(v_id)) {
      auto face_normal = mesh.faces_normals[f_id];

      of_z = 1 - std::abs(face_normal.x * stack_orientation[0] +
                          face_normal.y * stack_orientation[1] +
                          face_normal.z * stack_orientation[2]);
      of_y = 1 - std::abs(face_normal.x * 0 + 
                          face_normal.y * 1 +
                          face_normal.z * 0);
      of_x = 1 - std::abs(face_normal.x * 1 +
                          face_normal.y * 0 +
                          face_normal.z * 0);
      min_x = std::min(of_x, min_x);
      min_y = std::min(of_y, min_y);
      min_z = std::min(of_z, min_z);
      max_x = std::max(of_x, max_x);
      max_y = std::max(of_y, max_y);
      max_z = std::max(of_z, max_z);

      if (((std::abs(max_z - min_z) >= T) || (std::abs(max_y - min_y) >= T) ||
           (std::abs(max_x - min_x) >= T))) {
        output->push_back(v_id);
        break;
      }
    }
  }
  return output;
}

template <typename VERTICES_TYPE, typename ID_TYPE>
const std::shared_ptr<std::vector<float>>
calc_artifacts_weight(const MeshCPP<VERTICES_TYPE, ID_TYPE> &mesh,
                      const std::shared_ptr<std::vector<ID_TYPE>> vertices_staircase,
                      float tmax, float bmin) {
  auto weights = std::make_shared<std::vector<float>>(mesh.vertices.size());
  weights->assign(weights->size(), bmin);
  float distance;

  for (auto vi_id : *vertices_staircase) {
    auto vi = mesh.vertices[vi_id];
    auto near_vertices = mesh.get_near_vertices_to_v(vi_id, tmax);
    for (auto vj_id : *near_vertices) {
      auto vj = mesh.vertices[vj_id];
      distance = vertex_distance(vi, vj);
      float value = (1.0 - distance / tmax) * (1.0 - bmin) + bmin;
      (*weights)[vj_id] = std::max(value, (*weights)[vj_id]);
    }
  }

  return weights;
}

template <typename VERTICES_TYPE, typename ID_TYPE>
void ca_smoothing_cpp(MeshCPP<VERTICES_TYPE, ID_TYPE> &mesh, double T, double tmax,
                  double bmin, uint32_t n_iters) {
  std::array<double, 3> stack_orientation {0.0, 0.0, 1.0};
  auto t0 = std::chrono::system_clock::now();
  auto vertices_staircase =
      find_staircase_artifacts(mesh, stack_orientation, T);
  auto t1 = std::chrono::system_clock::now();
  std::cout << "1. Find artifacts: " << std::chrono::duration<double>(t1 - t0) << std::endl;

  std::cout << "Number of staircase artifacts: " << vertices_staircase->size() << std::endl;

  auto t2 = std::chrono::system_clock::now();
  auto weights =
      calc_artifacts_weight(mesh, vertices_staircase, tmax, bmin);
  auto t3 = std::chrono::system_clock::now();
  std::cout << "2. Calc weights: " << std::chrono::duration<double>(t3 - t2) << std::endl;
}

#endif

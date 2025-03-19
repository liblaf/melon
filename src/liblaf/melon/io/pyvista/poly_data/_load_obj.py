import os
from pathlib import Path

import attrs
import autoregistry
import numpy as np
import pyvista as pv

from liblaf import grapes


@attrs.define
class ParseState:
    current_group_id: int = -1
    current_object_id: int = -1
    face_vertex_normal_indices: list[list[int]] = attrs.field(factory=list)
    face_vertex_texture_coordinate_indices: list[list[int]] = attrs.field(factory=list)
    faces: list[list[int]] = attrs.field(factory=list)
    group_ids: list[int] = attrs.field(factory=list)
    group_names: list[str] = attrs.field(factory=list)
    object_ids: list[int] = attrs.field(factory=list)
    object_names: list[str] = attrs.field(factory=list)
    vertex_normals: list[list[float]] = attrs.field(factory=list)
    vertex_texture_coordinates: list[list[float]] = attrs.field(factory=list)
    vertices: list[list[float]] = attrs.field(factory=list)

    def get_or_create_object_id(self, name: str | None) -> int:
        if name is None:
            self.object_names.append(f"Object{len(self.object_names)}")
            return len(self.object_names) - 1
        try:
            return self.object_names.index(name)
        except ValueError:
            self.object_names.append(name)
            return len(self.object_names) - 1

    def get_or_create_group_id(self, name: str | None) -> int:
        if name is None:
            self.group_names.append(f"Group{len(self.group_names)}")
            return len(self.group_names) - 1
        try:
            return self.group_names.index(name)
        except ValueError:
            self.group_names.append(name)
            return len(self.group_names) - 1


def fix_index(index: int) -> int:
    return index - 1 if index > 0 else index


registry = autoregistry.Registry()


@registry
def v(state: ParseState, tokens: list[str]) -> ParseState:
    state.vertices.append([float(token) for token in tokens])
    return state


@registry
def vt(state: ParseState, tokens: list[str]) -> ParseState:
    state.vertex_texture_coordinates.append([float(token) for token in tokens])
    return state


@registry
def vn(state: ParseState, tokens: list[str]) -> ParseState:
    state.vertex_normals.append([float(token) for token in tokens])
    return state


@registry
def o(state: ParseState, tokens: list[str]) -> ParseState:
    state.current_object_id = state.get_or_create_object_id(tokens[0])
    return state


@registry
def g(state: ParseState, tokens: list[str]) -> ParseState:
    state.current_group_id = state.get_or_create_group_id(tokens[0])
    return state


@registry
def f(state: ParseState, tokens: list[str]) -> ParseState:
    state.group_ids.append(state.current_group_id)
    state.object_ids.append(state.current_object_id)
    v: list[int] = []
    vt: list[int] = []
    vn: list[int] = []
    for token in tokens:
        indices: list[str] = token.split("/")
        v.append(fix_index(int(indices[0])))
        if len(indices) >= 2:
            vt.append(fix_index(int(indices[1])))
        if len(indices) >= 3:
            vn.append(fix_index(int(indices[2])))
    state.faces.append(v)
    state.face_vertex_texture_coordinate_indices.append(vt)
    state.face_vertex_normal_indices.append(vn)
    return state


def load_obj(fpath: str | os.PathLike[str]) -> pv.PolyData:
    fpath: Path = grapes.as_path(fpath)
    text: str = fpath.read_text()
    state: ParseState = ParseState()
    for line in grapes.strip_comments(text):
        cmd: str
        tokens: list[str]
        cmd, *tokens = line.split()
        try:
            state = registry[cmd](state, tokens)
        except KeyError:
            grapes.warning_once(f"Unknown command: {cmd}")
    mesh: pv.PolyData = pv.PolyData.from_irregular_faces(state.vertices, state.faces)
    mesh.cell_data["GroupIds"] = np.asarray(state.group_ids)
    mesh.cell_data["ObjectIds"] = np.asarray(state.object_ids)
    mesh.field_data["GroupNames"] = state.group_names
    mesh.field_data["ObjectNames"] = state.object_names
    return mesh

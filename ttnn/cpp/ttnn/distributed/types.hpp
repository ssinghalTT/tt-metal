// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides type aliases for distributed computing components used in the TTNN library.
// It imports and renames types from the tt_metal library to maintain a consistent naming convention
// within the TTNN namespace while leveraging the underlying tt_metal functionality.

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
namespace ttnn::distributed {

using MeshShape = tt::tt_metal::distributed::MeshShape;
using MeshOffset = tt::tt_metal::distributed::MeshOffset;
using DeviceIds = tt::tt_metal::distributed::DeviceIds;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using SystemMesh = tt::tt_metal::distributed::SystemMesh;
using MeshDeviceView = tt::tt_metal::distributed::MeshDeviceView;
using MeshType = tt::tt_metal::distributed::MeshType;
using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;
using MeshSubDeviceManagerId = tt::tt_metal::distributed::MeshSubDeviceManagerId;

}  // namespace ttnn::distributed

namespace ttnn {

// These types are exported to the ttnn namespace for convenience.
using ttnn::distributed::DeviceIds;
using ttnn::distributed::MeshDevice;
using ttnn::distributed::MeshDeviceConfig;
using ttnn::distributed::MeshDeviceView;
using ttnn::distributed::MeshOffset;
using ttnn::distributed::MeshShape;
using ttnn::distributed::MeshSubDeviceManagerId;
using ttnn::distributed::MeshType;
using ttnn::distributed::SystemMesh;

}  // namespace ttnn

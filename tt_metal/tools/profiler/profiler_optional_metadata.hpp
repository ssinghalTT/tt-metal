#pragma once

#include "map"

class ProfilerOptionalMetadata {
    using DeviceID = uint32_t;
    using RuntimeID = uint32_t;

public:
    ProfilerOptionalMetadata(std::map<std::pair<DeviceID, RuntimeID>, std::string>&& runtime_map) :
        runtime_id_to_opname(std::move(runtime_map)) {}

    const std::string& getOpName(DeviceID device_id, RuntimeID runtime_id) const {
        static const std::string empty_string;
        auto key = std::make_pair(device_id, runtime_id);
        auto it = runtime_id_to_opname.find(key);
        if (it != runtime_id_to_opname.end()) {
            return it->second;
        }
        return empty_string;
    }

private:
    std::map<std::pair<DeviceID, RuntimeID>, std::string> runtime_id_to_opname;
};

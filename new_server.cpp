#include "ttnn/cpp/ttnn/multi_server/server.hpp"
#include "tt_metal/impl/device/device_mesh.hpp"
#include <cstdlib>
#include <string>

using namespace ttnn::multi_server;

int main() {
    const char* port = std::getenv("SERVER_PORT");
    if (!port) {
        port = "8086";  // Default port if not set
    }
    std::string address = "tcp://*:" + std::string(port);
    Worker<tt::tt_metal::DeviceMesh> worker(address);
    worker.run();
}

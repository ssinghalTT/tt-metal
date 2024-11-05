add_library(metal_api INTERFACE)
add_library(Metalium::Metal::API ALIAS metal_api)

# Con: feels a bit 'heavy' to have a target for the API that everyone depends on
target_include_directories(metal_api INTERFACE .)

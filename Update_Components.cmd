RD /s /q 3rd_party\Src\DirectXTex\DirectXTex
git clone --branch mar2025 --depth 1 https://github.com/microsoft/DirectXTex.git 3rd_party/Src/DirectXTex/DirectXTex

RD /s /q 3rd_party\Src\DirectXMesh\DirectXMesh
git clone --branch main --depth 1 https://github.com/OGSR/DirectXMesh.git 3rd_party/Src/DirectXMesh/DirectXMesh

RD /s /q 3rd_party\Src\DirectXMath\DirectXMath
git clone --branch main --depth 1 https://github.com/microsoft/DirectXMath.git 3rd_party/Src/DirectXMath/DirectXMath

RD /s /q 3rd_party\Src\concurrentqueue\concurrentqueue
git clone --branch master --depth 1 https://github.com/cameron314/concurrentqueue.git 3rd_party\Src\concurrentqueue\concurrentqueue

RD /s /q 3rd_party\Src\libsquashfs\squashfs-tools-ng
git clone --branch master --depth 1 https://github.com/AgentD/squashfs-tools-ng.git 3rd_party\Src\libsquashfs\squashfs-tools-ng

RD /s /q 3rd_party\Src\lz4\lz4
git clone --branch dev --depth 1 https://github.com/lz4/lz4.git 3rd_party\Src\lz4\lz4

RD /s /q 3rd_party\Src\zstd\zstd
git clone --branch dev --depth 1 https://github.com/facebook/zstd.git 3rd_party\Src\zstd\zstd

RD /s /q 3rd_party\Src\mimalloc\mimalloc
git clone --branch dev2 --depth 1 https://github.com/microsoft/mimalloc.git 3rd_party\Src\mimalloc\mimalloc

RD /s /q 3rd_party\Src\NVIDIA_DLSS\DLSS
git clone --branch v310.4.0 --depth 1 https://github.com/NVIDIA/DLSS.git 3rd_party\Src\NVIDIA_DLSS\DLSS

RD /s /q 3rd_party\Src\cpputils\cpputils
git clone --branch main --depth 1 https://github.com/tzcnt/cpputils.git 3rd_party\Src\cpputils\cpputils

REM ===== Vulkan renderer dependencies =====
REM VMA — header-only, used by xrRenderVulkan. Pinned to v3.4.x range (matches monolith).
RD /s /q 3rd_party\Src\VulkanMemoryAllocator\VulkanMemoryAllocator
git clone --branch master --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git 3rd_party\Src\VulkanMemoryAllocator\VulkanMemoryAllocator

REM NGX (DLSS for Vulkan) — currently *deferred*, kept commented so structure is ready.
REM Existing 3rd_party\Src\NVIDIA_DLSS\DLSS clone (above) is the same SDK used for Vulkan;
REM Vulkan renderer just needs a different lib variant from the same tree.
REM Reflex SDK — also deferred until requested by OGSR team.
REM RD /s /q 3rd_party\Src\NVIDIA_Reflex\Reflex
REM git clone --branch main --depth 1 https://github.com/NVIDIAGameWorks/Reflex.git 3rd_party\Src\NVIDIA_Reflex\Reflex

pause
// Socket headers for the S2 control path. g6m4 has a global namespace `bind` (kernels/bind.cuh), so POSIX bind() may not
// be declared in its translation unit: the headers are included with bind renamed, and bind goes through the syscall.
#pragma once
#define bind g6s2_posix_bind_undeclared
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#undef bind
#include <sys/syscall.h>
#include <unistd.h>

namespace g6s2 {
inline int posix_bind(int fd, const sockaddr* a, socklen_t n) { return int(::syscall(SYS_bind, fd, a, n)); }
} // namespace g6s2

#include "wrapper.h"

// Wrapper for inline functions

struct fi_info *wrap_fi_allocinfo(void) {
    return fi_allocinfo();
}

int wrap_fi_close(struct fid *fid) {
    return fi_close(fid);
}

int wrap_fi_endpoint(struct fid_domain *domain, struct fi_info *info,
                     struct fid_ep **ep, void *context) {
    return fi_endpoint(domain, info, ep, context);
}

int wrap_fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
                    struct fid_cq **cq, void *context) {
    return fi_cq_open(domain, attr, cq, context);
}

int wrap_fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
                    struct fid_av **av, void *context) {
    return fi_av_open(domain, attr, av, context);
}

int wrap_fi_ep_bind(struct fid_ep *ep, struct fid *bfid, uint64_t flags) {
    return fi_ep_bind(ep, bfid, flags);
}

int wrap_fi_enable(struct fid_ep *ep) {
    return fi_enable(ep);
}

ssize_t wrap_fi_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
                     fi_addr_t dest_addr, void *context) {
    return fi_send(ep, buf, len, desc, dest_addr, context);
}

ssize_t wrap_fi_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
                     fi_addr_t src_addr, void *context) {
    return fi_recv(ep, buf, len, desc, src_addr, context);
}

ssize_t wrap_fi_cq_read(struct fid_cq *cq, void *buf, size_t count) {
    return fi_cq_read(cq, buf, count);
}

ssize_t wrap_fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
                            fi_addr_t *src_addr) {
    return fi_cq_readfrom(cq, buf, count, src_addr);
}

ssize_t wrap_fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
                           uint64_t flags) {
    return fi_cq_readerr(cq, buf, flags);
}

const char *wrap_fi_cq_strerror(struct fid_cq *cq, int prov_errno,
                                const void *err_data, char *buf, size_t len) {
    return fi_cq_strerror(cq, prov_errno, err_data, buf, len);
}

int wrap_fi_getname(struct fid *fid, void *addr, size_t *addrlen) {
    return fi_getname(fid, addr, addrlen);
}

int wrap_fi_av_insert(struct fid_av *av, const void *addr, size_t count,
                      fi_addr_t *fi_addr, uint64_t flags, void *context) {
    return fi_av_insert(av, addr, count, fi_addr, flags, context);
}

int wrap_fi_domain(struct fid_fabric *fabric, struct fi_info *info,
                   struct fid_domain **domain, void *context) {
    return fi_domain(fabric, info, domain, context);
}

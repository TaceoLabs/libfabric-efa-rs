#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_collective.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

// Wrapper function declarations for inline functions
struct fi_info *wrap_fi_allocinfo(void);
int wrap_fi_close(struct fid *fid);
int wrap_fi_domain(struct fid_fabric *fabric, struct fi_info *info,
                   struct fid_domain **domain, void *context);
int wrap_fi_endpoint(struct fid_domain *domain, struct fi_info *info,
                     struct fid_ep **ep, void *context);
int wrap_fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
                    struct fid_cq **cq, void *context);
int wrap_fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
                    struct fid_av **av, void *context);
int wrap_fi_ep_bind(struct fid_ep *ep, struct fid *bfid, uint64_t flags);
int wrap_fi_enable(struct fid_ep *ep);
ssize_t wrap_fi_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
                     fi_addr_t dest_addr, void *context);
ssize_t wrap_fi_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
                     fi_addr_t src_addr, void *context);
ssize_t wrap_fi_cq_read(struct fid_cq *cq, void *buf, size_t count);
ssize_t wrap_fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
                            fi_addr_t *src_addr);
ssize_t wrap_fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
                           uint64_t flags);
const char *wrap_fi_cq_strerror(struct fid_cq *cq, int prov_errno,
                                const void *err_data, char *buf, size_t len);
int wrap_fi_getname(struct fid *fid, void *addr, size_t *addrlen);
int wrap_fi_av_insert(struct fid_av *av, const void *addr, size_t count,
                      fi_addr_t *fi_addr, uint64_t flags, void *context);

#endif /* __WRAPPER_H__ */

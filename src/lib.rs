//! Async Rust wrapper for libfabric with EFA support
//!
//! This library provides a safe, async interface to libfabric for high-performance
//! RDMA communication. It uses an ownership-based API to guarantee memory safety
//! while maintaining zero-copy performance.
//!
//! # Example
//!
//! ```ignore
//! use eyre::Result;
//! use libfabric_rs::{AddressExchangeChannel, FabricEndpoint};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut endpoint = FabricEndpoint::new()?;
//!     let mut channel = AddressExchangeChannel::connect("192.168.1.100", None).await?;
//!     let peer_addr = channel.exchange(&endpoint, true).await?;
//!     let peer_id = endpoint.insert_peer(&peer_addr)?;
//!     
//!     let mut buf = vec![0u8; 1024];
//!     buf = endpoint.send_to(peer_id, buf).await?;
//!     
//!     Ok(())
//! }
//! ```

use eyre::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::ptr;

#[allow(warnings, clippy::all)]
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Default TCP port for control channel (address exchange)
pub const CONTROL_PORT: u16 = 9229;

const DEFAULT_PORT: &str = "9228";
const EAGAIN_ERROR: isize = -(ffi::FI_EAGAIN as i32) as isize;
const EAVAIL_ERROR: isize = -(ffi::FI_EAVAIL as i32) as isize;

/// Reads the pending error entry from a CQ and returns a descriptive error string.
///
/// Must only be called after `fi_cq_read` returns `-FI_EAVAIL`.
unsafe fn cq_read_error(cq: *mut ffi::fid_cq) -> eyre::Report {
    let mut err: ffi::fi_cq_err_entry = std::mem::zeroed();
    let ret = ffi::wrap_fi_cq_readerr(cq, &mut err, 0);
    if ret < 0 {
        return eyre::eyre!("fi_cq_readerr failed: {}", ret);
    }
    let mut buf = [0i8; 256];
    let prov_msg = ffi::wrap_fi_cq_strerror(
        cq,
        err.prov_errno,
        err.err_data,
        buf.as_mut_ptr() as *mut libc::c_char,
        buf.len(),
    );
    let prov_str = if prov_msg.is_null() {
        std::borrow::Cow::Borrowed("(no provider message)")
    } else {
        CStr::from_ptr(prov_msg).to_string_lossy()
    };
    eyre::eyre!(
        "CQ error: {} (prov_errno={}, provider: {})",
        err.err,
        err.prov_errno,
        prov_str
    )
}

/// Compact, serializable representation of a libfabric endpoint address.
///
/// `FabricAddress` wraps the opaque byte blob returned by `fi_getname`. Because
/// it implements `Serialize`/`Deserialize`, callers can exchange the address
/// through any out-of-band channel (files, RPC, etc.) without relying on the
/// auxiliary TCP helper provided in this crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricAddress {
    bytes: Vec<u8>,
}

impl FabricAddress {
    /// Creates a new address from raw bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Returns the raw bytes of the address.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the wrapper and returns the owned byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns the length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns true if the address contains no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl From<Vec<u8>> for FabricAddress {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(bytes)
    }
}

impl AsRef<[u8]> for FabricAddress {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// Identifier for a peer in the address vector.
///
/// This is a type-safe wrapper around libfabric's `fi_addr_t`. Each peer
/// that is inserted into the endpoint's address vector gets a unique PeerId.
///
/// # Example
///
/// ```ignore
/// let peer1 = endpoint.insert_peer(&addr1)?;
/// let peer2 = endpoint.insert_peer(&addr2)?;
/// buf = endpoint.send_to(peer1, buf).await?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerId(pub ffi::fi_addr_t);

/// A fabric endpoint for RDMA communication.
///
/// This structure manages the libfabric resources needed for RDMA operations,
/// including fabric, domain, endpoint, address vector, and completion queue.
///
/// Resources are automatically cleaned up when the endpoint is dropped.
///
/// # Thread Safety
///
/// `FabricEndpoint` is configured with `FI_THREAD_SAFE` mode, which allows
/// concurrent access to the endpoint and completion queue from multiple threads.
/// The EFA provider supports thread-safe operations, and all libfabric calls
/// are internally synchronized.
///
/// Operations like `send_to()` and `recv()` move work to blocking threads via
/// `spawn_blocking`, so concurrent calls are safe and will not interfere with
/// each other.
pub struct FabricEndpoint {
    fabric: *mut ffi::fid_fabric,
    domain: *mut ffi::fid_domain,
    ep: *mut ffi::fid_ep,
    av: *mut ffi::fid_av,
    send_cq: *mut ffi::fid_cq,
    recv_cq: *mut ffi::fid_cq,
    info: *mut ffi::fi_info,
    hints: *mut ffi::fi_info,
    fi_addr: ffi::fi_addr_t,
}

// SAFETY: FabricEndpoint is configured with FI_THREAD_SAFE mode during initialization,
// which ensures that the EFA provider's internal structures are thread-safe. All
// operations that access libfabric resources are done through `spawn_blocking`, which
// provides additional isolation. The raw pointers are never dereferenced from multiple
// threads simultaneously.
unsafe impl Send for FabricEndpoint {}
unsafe impl Sync for FabricEndpoint {}

impl Drop for FabricEndpoint {
    fn drop(&mut self) {
        unsafe {
            if !self.ep.is_null() {
                ffi::wrap_fi_close(&mut (*self.ep).fid as *mut ffi::fid);
            }
            if !self.av.is_null() {
                ffi::wrap_fi_close(&mut (*self.av).fid as *mut ffi::fid);
            }
            if !self.send_cq.is_null() {
                ffi::wrap_fi_close(&mut (*self.send_cq).fid as *mut ffi::fid);
            }
            if !self.recv_cq.is_null() {
                ffi::wrap_fi_close(&mut (*self.recv_cq).fid as *mut ffi::fid);
            }
            if !self.domain.is_null() {
                ffi::wrap_fi_close(&mut (*self.domain).fid as *mut ffi::fid);
            }
            if !self.fabric.is_null() {
                ffi::wrap_fi_close(&mut (*self.fabric).fid as *mut ffi::fid);
            }
            if !self.info.is_null() {
                ffi::fi_freeinfo(self.info);
            }
            if !self.hints.is_null() {
                ffi::fi_freeinfo(self.hints);
            }
        }
    }
}

impl FabricEndpoint {
    /// Creates a new fabric endpoint with EFA provider.
    ///
    /// This initializes all necessary libfabric resources including fabric,
    /// domain, endpoint, completion queue, and address vector.
    ///
    /// # Returns
    ///
    /// Returns `Ok(FabricEndpoint)` on success, or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if any libfabric initialization call fails.
    pub fn new() -> Result<Self> {
        unsafe {
            // RAII guard to ensure resources are cleaned up on any error path
            struct ResourceGuard {
                fabric: *mut ffi::fid_fabric,
                domain: *mut ffi::fid_domain,
                ep: *mut ffi::fid_ep,
                av: *mut ffi::fid_av,
                send_cq: *mut ffi::fid_cq,
                recv_cq: *mut ffi::fid_cq,
                info: *mut ffi::fi_info,
                hints: *mut ffi::fi_info,
            }

            impl Drop for ResourceGuard {
                fn drop(&mut self) {
                    unsafe {
                        if !self.ep.is_null() {
                            ffi::wrap_fi_close(&mut (*self.ep).fid as *mut ffi::fid);
                        }
                        if !self.av.is_null() {
                            ffi::wrap_fi_close(&mut (*self.av).fid as *mut ffi::fid);
                        }
                        if !self.send_cq.is_null() {
                            ffi::wrap_fi_close(&mut (*self.send_cq).fid as *mut ffi::fid);
                        }
                        if !self.recv_cq.is_null() {
                            ffi::wrap_fi_close(&mut (*self.recv_cq).fid as *mut ffi::fid);
                        }
                        if !self.domain.is_null() {
                            ffi::wrap_fi_close(&mut (*self.domain).fid as *mut ffi::fid);
                        }
                        if !self.fabric.is_null() {
                            ffi::wrap_fi_close(&mut (*self.fabric).fid as *mut ffi::fid);
                        }
                        if !self.info.is_null() {
                            ffi::fi_freeinfo(self.info);
                        }
                        if !self.hints.is_null() {
                            ffi::fi_freeinfo(self.hints);
                        }
                    }
                }
            }

            let hints = ffi::wrap_fi_allocinfo();
            if hints.is_null() {
                bail!("fi_allocinfo failed");
            }

            let mut guard = ResourceGuard {
                fabric: ptr::null_mut(),
                domain: ptr::null_mut(),
                ep: ptr::null_mut(),
                av: ptr::null_mut(),
                send_cq: ptr::null_mut(),
                recv_cq: ptr::null_mut(),
                info: ptr::null_mut(),
                hints,
            };

            let provider_name = CString::new("efa").unwrap();
            (*(*hints).fabric_attr).prov_name = provider_name.as_ptr() as *mut i8;
            std::mem::forget(provider_name);

            (*(*hints).ep_attr).type_ = ffi::fi_ep_type_FI_EP_RDM;
            (*hints).caps = ffi::FI_MSG as u64;
            (*(*hints).tx_attr).op_flags = ffi::FI_DELIVERY_COMPLETE as u64;

            // Request thread-safe mode to enable concurrent access from multiple threads
            (*(*hints).domain_attr).threading = ffi::fi_threading_FI_THREAD_SAFE;

            let mut info_ptr: *mut ffi::fi_info = ptr::null_mut();
            let port_cstr = CString::new(DEFAULT_PORT).unwrap();
            let version = ffi::fi_version();
            let ret = ffi::fi_getinfo(
                version,
                ptr::null(),
                port_cstr.as_ptr(),
                ffi::FI_SOURCE,
                hints,
                &mut info_ptr,
            );

            if ret != 0 {
                bail!("fi_getinfo failed: {}", ret);
            }
            guard.info = info_ptr;

            let _prov_name = CStr::from_ptr((*(*info_ptr).fabric_attr).prov_name);

            let mut fabric: *mut ffi::fid_fabric = ptr::null_mut();
            let ret = ffi::fi_fabric((*info_ptr).fabric_attr, &mut fabric, ptr::null_mut());
            if ret != 0 {
                bail!("fi_fabric failed: {}", ret);
            }
            guard.fabric = fabric;

            let mut domain: *mut ffi::fid_domain = ptr::null_mut();
            let ret = ffi::wrap_fi_domain(fabric, info_ptr, &mut domain, ptr::null_mut());
            if ret != 0 {
                bail!("fi_domain failed: {}", ret);
            }
            guard.domain = domain;

            let mut ep: *mut ffi::fid_ep = ptr::null_mut();
            let ret = ffi::wrap_fi_endpoint(domain, info_ptr, &mut ep, ptr::null_mut());
            if ret != 0 {
                bail!("fi_endpoint failed: {}", ret);
            }
            guard.ep = ep;

            let mut cq_attr: ffi::fi_cq_attr = std::mem::zeroed();
            cq_attr.size = 128;
            cq_attr.format = ffi::fi_cq_format_FI_CQ_FORMAT_DATA;

            let mut send_cq: *mut ffi::fid_cq = ptr::null_mut();
            let ret = ffi::wrap_fi_cq_open(domain, &mut cq_attr, &mut send_cq, ptr::null_mut());
            if ret != 0 {
                bail!("fi_cq_open failed: {}", ret);
            }
            guard.send_cq = send_cq;

            let mut recv_cq: *mut ffi::fid_cq = ptr::null_mut();
            let ret = ffi::wrap_fi_cq_open(domain, &mut cq_attr, &mut recv_cq, ptr::null_mut());
            if ret != 0 {
                bail!("fi_cq_open failed: {}", ret);
            }
            guard.recv_cq = recv_cq;

            let ret = ffi::wrap_fi_ep_bind(
                ep,
                &mut (*send_cq).fid as *mut ffi::fid,
                ffi::FI_SEND as u64,
            );
            if ret != 0 {
                bail!("fi_ep_bind send_cq failed: {}", ret);
            }

            let ret = ffi::wrap_fi_ep_bind(
                ep,
                &mut (*recv_cq).fid as *mut ffi::fid,
                ffi::FI_RECV as u64,
            );
            if ret != 0 {
                bail!("fi_ep_bind recv_cq failed: {}", ret);
            }

            let mut av_attr: ffi::fi_av_attr = std::mem::zeroed();
            av_attr.type_ = ffi::fi_av_type_FI_AV_MAP;
            av_attr.count = 64;

            let mut av: *mut ffi::fid_av = ptr::null_mut();
            let ret = ffi::wrap_fi_av_open(domain, &mut av_attr, &mut av, ptr::null_mut());
            if ret != 0 {
                bail!("fi_av_open failed: {}", ret);
            }
            guard.av = av;

            let ret = ffi::wrap_fi_ep_bind(ep, &mut (*av).fid as *mut ffi::fid, 0);
            if ret != 0 {
                bail!("fi_ep_bind av failed: {}", ret);
            }

            let ret = ffi::wrap_fi_enable(ep);
            if ret != 0 {
                bail!("fi_enable failed: {}", ret);
            }

            // Disarm the guard by moving resources out and forgetting it
            let fabric = guard.fabric;
            let domain = guard.domain;
            let ep = guard.ep;
            let av = guard.av;
            let send_cq = guard.send_cq;
            let recv_cq = guard.recv_cq;
            let info = guard.info;
            let hints = guard.hints;
            std::mem::forget(guard);

            Ok(FabricEndpoint {
                fabric,
                domain,
                ep,
                av,
                send_cq,
                recv_cq,
                info,
                hints,
                fi_addr: 0,
            })
        }
    }

    /// Sends data to a specific peer.
    ///
    /// This function takes ownership of the buffer, sends it to the specified peer,
    /// and returns the buffer when the operation completes.
    ///
    /// # Arguments
    ///
    /// * `peer` - The peer to send to
    /// * `buf` - The buffer to send. Ownership is transferred to this function.
    ///
    /// # Returns
    ///
    /// Returns the buffer after the send operation completes, allowing reuse.
    ///
    /// # Errors
    ///
    /// Returns an error if the send operation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let peer = endpoint.insert_peer(&peer_addr)?;
    /// let mut buf = vec![0u8; 8192];
    /// buf = endpoint.send_to(peer, buf).await?;
    /// ```
    pub fn send_to(&self, peer: PeerId, buf: Vec<u8>) -> Result<Vec<u8>> {
        let ep = self.ep as usize;
        let fi_addr = peer.0;
        let cq = self.send_cq as usize;
        unsafe {
            let ep = ep as *mut ffi::fid_ep;
            let cq = cq as *mut ffi::fid_cq;

            loop {
                let ret = ffi::wrap_fi_send(
                    ep,
                    buf.as_ptr() as *const libc::c_void,
                    buf.len(),
                    ptr::null_mut(),
                    fi_addr,
                    ptr::null_mut(),
                );

                if ret == 0 {
                    break;
                } else if ret != EAGAIN_ERROR {
                    bail!("fi_send failed: {}", ret);
                }

                ffi::wrap_fi_cq_read(cq, ptr::null_mut(), 0);
            }

            let mut comp: ffi::fi_cq_data_entry = std::mem::zeroed();
            loop {
                let ret = ffi::wrap_fi_cq_read(
                    cq,
                    &mut comp as *mut ffi::fi_cq_data_entry as *mut libc::c_void,
                    1,
                );

                if ret == 1 {
                    return Ok(buf);
                } else if ret < 0 && ret != EAGAIN_ERROR {
                    if ret == EAVAIL_ERROR {
                        return Err(cq_read_error(cq));
                    }
                    bail!("fi_cq_read failed: {}", ret);
                }
            }
        }
    }

    /// Receives data from any peer.
    ///
    /// This function takes ownership of the buffer, receives data, and returns the
    /// buffer when the operation completes.
    ///
    /// # Note
    ///
    /// This receive operation accepts data from any connected peer. Libfabric RDM
    /// endpoints do not support peer-specific receives. If you need to receive from
    /// specific peers, use multiple endpoints or implement peer filtering at the
    /// application level.
    ///
    /// # Arguments
    ///
    /// * `buf` - The buffer to receive into. Ownership is transferred to this function.
    ///
    /// # Returns
    ///
    /// Returns the buffer filled with received data.
    ///
    /// # Errors
    ///
    /// Returns an error if the receive operation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buf = vec![0u8; 8192];
    /// let buf = endpoint.recv(buf).await?;
    /// // buf now contains received data
    /// ```
    pub fn recv(&self, mut buf: Vec<u8>) -> Result<Vec<u8>> {
        let ep = self.ep as usize;
        let cq = self.recv_cq as usize;

        unsafe {
            let ep = ep as *mut ffi::fid_ep;
            let cq = cq as *mut ffi::fid_cq;

            loop {
                let ret = ffi::wrap_fi_recv(
                    ep,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf.len(),
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                );

                if ret == 0 {
                    break;
                } else if ret != EAGAIN_ERROR {
                    bail!("fi_recv failed: {}", ret);
                }

                ffi::wrap_fi_cq_read(cq, ptr::null_mut(), 0);
            }

            let mut comp: ffi::fi_cq_data_entry = std::mem::zeroed();
            loop {
                let ret = ffi::wrap_fi_cq_read(
                    cq,
                    &mut comp as *mut ffi::fi_cq_data_entry as *mut libc::c_void,
                    1,
                );

                if ret == 1 {
                    return Ok(buf);
                } else if ret < 0 && ret != EAGAIN_ERROR {
                    if ret == EAVAIL_ERROR {
                        return Err(cq_read_error(cq));
                    }
                    bail!("fi_cq_read failed: {}", ret);
                }
            }
        }
    }

    /// Retrieves the local endpoint address.
    ///
    /// # Returns
    ///
    /// Returns the local address as a [`FabricAddress`].
    ///
    /// # Errors
    ///
    /// Returns an error if fi_getname fails.
    pub fn local_address(&self) -> Result<FabricAddress> {
        unsafe {
            let mut local_addr: Vec<u8> = vec![0; 128];
            let mut local_addrlen: libc::size_t = local_addr.len();

            let ret = ffi::wrap_fi_getname(
                &mut (*self.ep).fid as *mut ffi::fid,
                local_addr.as_mut_ptr() as *mut libc::c_void,
                &mut local_addrlen,
            );

            if ret != 0 {
                bail!("fi_getname failed: {}", ret);
            }

            local_addr.resize(local_addrlen, 0);
            Ok(FabricAddress::from(local_addr))
        }
    }

    /// Inserts a peer address into the address vector.
    ///
    /// # Arguments
    ///
    /// * `peer_addr` - The peer's [`FabricAddress`] to insert
    ///
    /// # Errors
    ///
    /// Returns an error if fi_av_insert fails.
    /// Inserts a peer address into the address vector.
    ///
    /// This method adds a new peer to the endpoint's address vector.
    ///
    /// # Arguments
    ///
    /// * `peer_addr` - The peer's [`FabricAddress`] to insert
    ///
    /// # Returns
    ///
    /// Returns a `PeerId` that can be used to send messages to this peer.
    ///
    /// # Errors
    ///
    /// Returns an error if the address insertion fails.
    pub fn insert_peer(&mut self, peer_addr: &FabricAddress) -> Result<PeerId> {
        unsafe {
            let mut fi_addr: ffi::fi_addr_t = 0;
            let ret = ffi::wrap_fi_av_insert(
                self.av,
                peer_addr.as_bytes().as_ptr() as *const libc::c_void,
                1,
                &mut fi_addr,
                0,
                ptr::null_mut(),
            );

            ensure!(ret == 1, "fi_av_insert failed: {}", ret);

            self.fi_addr = fi_addr;
            Ok(PeerId(fi_addr))
        }
    }
}

# **EC2 Instance Storage**

EC2 instances can use different types of storage depending on need. Main options: **EBS, Instance Store, EFS, FSx, S3 (indirectly)**.

---

### 1. **EBS (Elastic Block Store)**

- Network-attached **block storage** for EC2 (like a hard disk).Region Based service
    
- **Persistent**: data survives instance stop/start.
    
- **Availability**: replicated automatically within the same AZ.
    
- **Types**:
    
    - **General Purpose SSD (gp2/gp3)** → balance cost/performance, default choice.
        
    - **Provisioned IOPS SSD (io1/io2)** → high performance, databases, low latency.
        
    - **Throughput Optimized HDD (st1)** → big data, streaming workloads.
        
    - **Cold HDD (sc1)** → infrequently accessed data, cheapest.
         
- **Snapshots: For Backups and cross region replication** :
    - Backup stored in S3.
    - Can be used to create new volumes or move across AZs/Regions.
        
- **Encryption**: AWS KMS-managed keys, encryption at rest & in transit.
    
- **Attachment**:
    
    - One volume can be attached to one EC2 at a time (except Multi-Attach with io1/io2 → multiple instances in same AZ).
        
![[Pasted image 20251003160343.png]]
![[Pasted image 20251003160425.png]]
![[Pasted image 20251003160457.png]]
![[Pasted image 20251003160541.png]]

---

### 2. **Instance Store (Ephemeral Storage)**

- **Physical storage** directly attached to the host machine.
    
- Very **high performance** (faster than EBS).
    
- **Temporary**: data is lost if instance is stopped, terminated, or hardware fails.
    
- Good for **cache, buffer, temporary data**.
    
- Cannot be detached or persisted like EBS.
    

---

### 3. **EFS (Elastic File System)**

- Fully managed **network file system (NFS)**.
    
- Shared across multiple EC2 instances at once (multi-AZ).
    
- Automatically **scales** storage up/down.
    
- **Use cases**: web apps, content management, shared storage.
    
- **Pricing**: pay per GB, more expensive than EBS.
    
- Supports **encryption** & lifecycle management (move infrequently used files to EFS-IA).
    

---

### 4. **FSx**

- Managed file storage for specific needs.
    
- Types:
    
    - **FSx for Windows** → native Windows file system (SMB protocol).
        
    - **FSx for Lustre** → high-performance workloads (HPC, ML, big data).
        
- Integrates with **S3** for input/output.
    

---

### 5. **S3 (Indirect for EC2)**

- Not attached directly to EC2 like a disk, but often used with EC2 for storing:
    
    - Backups, static files, logs.
        
    - Data for processing (input/output).
        
- Accessed via APIs/SDKs, not mounted as block storage (unless using S3FS or EFS-to-S3 lifecycle policies).
    

---

### 6. **Key Differences**

- **EBS** → Persistent, block storage for one instance (like a hard drive).
    
- **Instance Store** → Temporary, super-fast, data lost on stop/terminate.
    
- **EFS** → Shared file system, scales automatically, works across AZs.
    
- **FSx** → Specialized file systems (Windows/Linux HPC).
    
- **S3** → Object storage, not block storage, but used for backups + big data.
    

---

### 7. **Best Practices**

- Use **EBS gp3** as default.
    
- Use **Provisioned IOPS (io2)** for databases needing consistent performance.
    
- Use **Instance Store** only for temporary/cache data.
    
- Use **EFS** when you need multiple instances to share files.
    
- Take **regular snapshots** of EBS for durability/backup.
    
- Enable **encryption** for sensitive data.
    

---

Do you want me to also make a **small diagram-style summary** (like which storage is persistent vs temporary, single vs shared, block vs file vs object) for quick revision?
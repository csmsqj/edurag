import os
import sys
absfile=os.path.abspath(__file__)
rootfile=os.path.dirname(os.path.dirname(os.path.dirname(absfile)))
dirfile=os.path.dirname(absfile)
sys.path.insert(0,dirfile)
sys.path.insert(0,rootfile)
#导入 BGE-M3 嵌入函数（来自 milvus-model 包）
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 客户端和相关类型
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 LangChain 的 Document 类型
from langchain_core.documents import Document
# 导入 CrossEncoder（用于重排序）
from sentence_transformers import CrossEncoder
# 导入哈希库（用于生成文档指纹，去重）
import hashlib
from base.config import Config
from base.log import get_logger
class VectorStore:
    """向量存储类，封装了向量数据库的所有操作"""
    def __init__(self,
                 collection_name=None,
                 host=None,
                 port=None,
                 database=None):
        self.config=Config()
        if collection_name is None:
            collection_name=self.config.COLLECTION_NAME
        if host is None:
            host=self.config.MILVUS_HOST
        if port is None:
            port=self.config.MILVUS_PORT
        if database is None:
            database=self.config.MILVUS_DATABASE

        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口
        self.port = port
        # 设置数据库名称
        self.database = database
        # 日志记录器
        self.logger = get_logger()
        file_reranker=os.path.join(rootfile,'models/bge-reranker-large')
        file_bge=os.path.join(rootfile,'models/bge-m3')
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        self.reranker = CrossEncoder(file_reranker)

        # 初始化 BGE-M3 嵌入模型
        # use_fp16=False: 不用半精度，用全精度（FP32）
        # device='cpu': 在 CPU 上运行（没有 GPU 时）
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name=file_bge,  # 本地模型路径，从 config.ini 的 [local] m3_model 读取
            use_fp16=True,
            device='cuda'#gpu
        )

        # 获取稠密向量的维度（1024）
        self.dense_dim = self.embedding_function.dim["dense"]

        # 先连接默认库，确保目标数据库存在
        temp_client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        if self.database not in temp_client.list_databases():
            temp_client.create_database(self.database)
        temp_client.close()
        # 初始化 Milvus 客户端，连接到 Milvus 服务器
        self.client = MilvusClient(
            uri=f"http://{self.host}:{self.port}",
            db_name=self.database
        )
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()


    def _create_or_load_collection(self):
        """
        创建或加载 Milvus 集合
        如果集合已经存在，直接加载；不存在就创建一个新的。
        """
        if self.client.has_collection(self.collection_name):
            # 集合已存在，直接加载到内存（Milvus 需要先 load 才能搜索）
            self.client.load_collection(self.collection_name)
            self.logger.info(f"Milvus 集合 '{self.collection_name}' 已存在，已加载。")
        else:
            self.logger.info(f"Milvus 集合 '{self.collection_name}' 不存在，创建...")
            # 第一步：定义 Schema（表结构）
            schema = self.client.create_schema(
                auto_id=False,  # 主键 ID 手动传
                enable_dynamic_field=True  # 允许动态字段——Schema 里没定义的字段也能存
                # 这样 parent_id、parent_content 等元数据
                # 不需要在 Schema 里显式声明也能存进去
            )

            # 第二步：添加字段
            schema.add_field(field_name="id", datatype=DataType.VARCHAR,max_length=1024,
                             is_primary=True)  # 主键，自动生成
            schema.add_field(field_name="content", datatype=DataType.VARCHAR,
                             max_length=65535)  # 子块的文本内容
            schema.add_field(field_name="sparse_vector",
                             datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量（维度不固定）
            schema.add_field(field_name="dense_vector",
                             datatype=DataType.FLOAT_VECTOR,
                             dim=self.dense_dim)  # 稠密向量，1024 维

            # 第三步：创建索引（没有索引，搜索会很慢）
            index_params = self.client.prepare_index_params()

            # 稠密向量的索引
            index_params.add_index(
                field_name="dense_vector",
                index_type="IVF_FLAT",  # 倒排文件 + 平面量化（精度高、速度适中）
                metric_type="COSINE",  # 用余弦相似度衡量距离
                params={"nlist": 128}  # 把向量聚成 128 个簇，查询时可以选择只搜最近的几个簇
            )

            # 稀疏向量的索引
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",  # 稀疏向量专用的倒排索引
                metric_type="IP" , # 用内积（Inner Product）衡量相似度
            drop_ratio_build = 0.1# 构建索引时丢弃掉 10% 的最稀疏向量，以提高查询效率（可选参数）
            )

            # 第四步：创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

            self.logger.info(f"创建新集合：{self.collection_name}")


    #子块列表documents,添加到 Milvus 集合
    #:param documents: List[Document] - 要添加的文档列表
    def add_documents(self, documents):
        """
                把 process_documents.py 输出的子块列表存进 Milvus。
                流程：
                1. 提取所有子块的文本
                2. 用 BGE-M3 一次性编码成 稠密向量 + 稀疏向量
                3. 拼装数据（向量 + 文本 + 元数据）
                4. 批量插入 Milvus
        """
        if documents is None or len(documents) == 0:
            self.logger.warning("没有文档可添加到 Milvus。")
            return
        # 1. 提取所有子块的文本
        test_list=[]
        for doc in documents:
            test_list.append(doc.page_content)
        # 2. 用 BGE-M3 一次性编码成 稠密向量 + 稀疏向量
        # 创建的 BGEM3EmbeddingFunction 对象
        # 传入：字符串列表（所有子块的文本）
        # 返回：一个字典，包含两种向量：
        #   embeddings["dense"]  → 稠密向量数组，每个文本一个 1024 维的向量（抓语义）
        #   embeddings["sparse"] → 稀疏向量数组，每个文本一个变长的向量（抓关键词）
        embeddings = self.embedding_function(test_list)
        # 3.去重准备：批量查询 Milvus 中已存在的 content_hash
        #遍历 documents 列表，提取每个 Document 的 page_content，计算哈希值，存到字典列表，再插入 Milvus
        milvus_list=[]
        for i,doc in enumerate(documents):
            # 生成 content_hash：对 page_content 进行 MD5 哈希，得到一个固定长度的字符串，作为文档指纹
            content_hash = hashlib.md5(
                doc.page_content.encode('utf-8')
            ).hexdigest()
            #包含mivlus字段，和动态字段parent_id、parent_content,source
            ment={
                "id":content_hash,
                "content": doc.page_content,
                "sparse_vector": embeddings["sparse"][[i]],
                "dense_vector": embeddings["dense"][i].tolist(),  # 转成列表，Milvus 要求

                "parent_id": doc.metadata.get("parent_id", ""),
                "parent_content": doc.metadata.get("parent_content", ""),
                "source": doc.metadata.get("source", "")

            }
            milvus_list.append(ment)
        # 4. 批量插入 Milvus
        self.client.upsert(
            collection_name=self.collection_name,
                           data=milvus_list
                           )
        self.logger.info(f"成功添加 {len(milvus_list)} 个文档到 Milvus 集合 '{self.collection_name}'。")


    # 从 Milvus 检索相关文档的函数
    def search(self, query, top_k_bgem3=5,top_k_rank=3):
        #1. 对查询语句进行编码，得到查询的稠密向量和稀疏向量
        query_bge=self.embedding_function([query])
        query_dense=query_bge.get("dense")[0].tolist()
        raw_sparse = query_bge.get("sparse")  # csr_array, shape (1, vocab_size)
        query_sparse = [{int(idx): float(val) for idx, val in zip(raw_sparse.indices, raw_sparse.data)}]

        # ===== 第二步：构造两个检索请求 =====
        # Dense 检索（语义匹配）
        dense_search = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k_bgem3
        )

        # Sparse 检索（关键词匹配）
        sparse_search = AnnSearchRequest(
            data=query_sparse,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio": 0.1}},
            limit=top_k_bgem3
        )

        # ===== 第三步：执行检索 =====
        # ===== 第三步：执行混合检索 =====
        # WeightedRanker(0.7, 0.3)：Dense 权重 70%，Sparse 权重 30%
        # 为什么 Dense 权重更高？因为语义理解通常比关键词匹配更重要
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_search, sparse_search],
            ranker=WeightedRanker(0.7, 0.3),
            limit=top_k_bgem3,
            output_fields=["content", "parent_id", "parent_content", "source"]
        )
        #RESULTS 为查几个内容就对应几个列表，列表当中是一个个字典，每个字典对应一个查询结果(就这样一个结果为一个字典，多个结果就是多个字典)
        #然后这个字典有 ID 键值对，距离键值对和 ENTITY 键值对。ENTITY 键值对是输出字段组成的字典。

            # 检查检索结果是否为空
        if not results or not results[0]:
            return []
        # ===== 第四步：CrossEncoder 重排序 =====
        candidates = []
        for hit in results[0]:
            candidates.append({
                "content": hit["entity"].get("content",""),  # 子块文本内容
                "parent_id": hit["entity"].get("parent_id", ""),  # 父块 ID
                "parent_content": hit["entity"].get("parent_content", ""),  # 父块完整文本
                "source": hit["entity"].get("source", ""),  # 来源分类
                "distance": hit.get("distance", 0)  # Milvus 返回的原始距离分数
            })
        # pairs 是一个二维列表，每个元素是 [query, 候选文本]
        # 例如：pairs = [["Redis怎么用", "Redis是内存数据库..."], ["Redis怎么用", "Spring Boot是..."]]
        # CrossEncoder 会把每对 [query, doc] 拼接后输入 BERT，输出一个相关性分数
        pairs=[]
        for c in candidates:
            pairs.append([query,c["content"]])
        rerank_scores=self.reranker.predict(pairs)
        # 把原来的 candidates 和 rerank_scores 结合起来，得到最终的重
        # 把 CrossEncoder 分数写回每个候选的字典里
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)
        # 根据 CrossEncoder 分数重新排序，分数高的排在前面
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        # 获取top_k_rank 个相关文档
        top_candidates = candidates[:top_k_rank]
        # 回溯父块并去重：多个子块可能属于同一个父块，只保留一份父块内容给大模型
        final_documents = self._get_unique_parent_documents(top_candidates)
        #返回要给大模型的文档列表。它是由父块内容、父块 ID ，对应资源，还有两个模型评分来组成的
        return final_documents

#去重函数，用于从候选子块中获取唯一的父块文档，为search函数服务
    def _get_unique_parent_documents(self, candidates):
        """
        根据候选子块，取回唯一的父块文档。
        去重逻辑：如果多个子块属于同一个父块，只返回一次父块内容。
        """
        seen_parent_ids = set()   # 已经见过的父块 ID
        final_documents = []

        for candidate in candidates:
            parent_id = candidate.get("parent_id", "")

            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)

                # 用父块的完整内容（512字）而不是子块的内容（128字）
                content = candidate.get("parent_content", candidate["content"])
                final_documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": candidate.get("source", ""),
                        "parent_id": parent_id,
                        "rerank_score": candidate.get("rerank_score", 0),
                        "distance": candidate.get("distance", 0)
                    }
                ))
        return final_documents





if __name__     == '__main__':
    vector_store = VectorStore()
    from document_loader_splitter import document_loader_splitter
    loader_splitter=document_loader_splitter()
    directory_path=os.path.join(rootfile,'data')
    documents=loader_splitter.process_documents(directory_path)
    #其他地方直接搜索就可以，这里先把文档添加到 Milvus 集合里
    vector_store.add_documents(documents)
    query=("java任课老师是谁？")
    results=vector_store.search(query)
    for i,res in enumerate(results):
        print(f"最终结果 {i+1}:{res.page_content}，来源：{res.metadata.get('source','')},重排序分数：{res.metadata.get('rerank_score',0)}")



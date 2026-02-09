from sklearn.metrics.pairwise import euclidean_distances
import heapq


# util.py의 nun 함수 수정
def nun(data, labels, query, query_label, cf_label):
    sample_size = len(data)
    ecd = euclidean_distances([query], data)[0]
    top_indices = heapq.nsmallest(sample_size, range(len(ecd)), ecd.take)
    
    # [수정] 정확한 일치가 아니라, 현재보다 점수가 낮은 이웃을 찾음
    for i in top_indices:
        lab = labels[i]
        # 현재 스트레스 점수(query_label)보다 낮은 데이터를 찾으면 바로 반환
        if lab < query_label: 
            return data[i], lab
            
    raise Exception('NUN not found. 현재보다 낮은 점수를 가진 데이터가 없습니다.')

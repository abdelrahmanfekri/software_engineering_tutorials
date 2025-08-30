## 15 — Advanced Data Structures

Coordinate compression
```java
import java.util.*;

public class Compress {
  public static int[] compress(int[] arr) {
    int[] sorted = arr.clone(); 
    Arrays.sort(sorted);
    int m = 0; 
    for (int i = 0; i < sorted.length; i++) 
    if (i==0 || sorted[i]!=sorted[i-1]) 
    sorted[m++] = sorted[i];
    int[] comp = new int[arr.length];
    for (int i = 0; i < arr.length; i++) 
    comp[i] = Arrays.binarySearch(sorted, 0, m, arr[i]);
    return comp;
  }
}
```

Segment Tree with Lazy Propagation (range add, range sum)
```java
public class SegmentTreeLazy {
  private final int n; private final long[] tree, lazy;
  public SegmentTreeLazy(int size){ int s=1; while(s<size) s<<=1; n=s; tree=new long[n<<1]; lazy=new long[n<<1]; }
  private void apply(int idx,long val,int len){ tree[idx]+=val*len; lazy[idx]+=val; }
  private void push(int idx,int len){ if(lazy[idx]!=0){ apply(idx<<1, lazy[idx], len>>1); apply(idx<<1|1, lazy[idx], len>>1); lazy[idx]=0; } }
  public void rangeAdd(int l,int r,long val){ rangeAdd(l,r,val,1,0,n, n); }
  private void rangeAdd(int l,int r,long val,int idx,int L,int R,int len){ if(r<=L||R<=l) return; if(l<=L&&R<=r){ apply(idx,val,R-L); return; } push(idx, R-L); int M=(L+R)>>1; rangeAdd(l,r,val,idx<<1,L,M,len>>1); rangeAdd(l,r,val,idx<<1|1,M,R,len>>1); tree[idx]=tree[idx<<1]+tree[idx<<1|1]; }
  public long rangeSum(int l,int r){ return rangeSum(l,r,1,0,n); }
  private long rangeSum(int l,int r,int idx,int L,int R){ if(r<=L||R<=l) return 0; if(l<=L&&R<=r) return tree[idx]; push(idx, R-L); int M=(L+R)>>1; return rangeSum(l,r,idx<<1,L,M)+rangeSum(l,r,idx<<1|1,M,R); }
}
```

Trie (prefix tree)
```java
public class Trie {
  static class Node { Node[] next = new Node[26]; boolean end; }
  private final Node root = new Node();
  public void insert(String s){ Node cur=root; for(char ch: s.toCharArray()){ int i=ch-'a'; if(cur.next[i]==null) cur.next[i]=new Node(); cur=cur.next[i]; } cur.end=true; }
  public boolean search(String s){ Node cur=root; for(char ch: s.toCharArray()){ int i=ch-'a'; if(cur.next[i]==null) return false; cur=cur.next[i]; } return cur.end; }
  public boolean startsWith(String p){ Node cur=root; for(char ch: p.toCharArray()){ int i=ch-'a'; if(cur.next[i]==null) return false; cur=cur.next[i]; } return true; }
}
```

Order statistics via Fenwick and compression
```java
public class KthOrderStatistics {
  static class Fenwick { long[] bit; int n; Fenwick(int n){this.n=n; bit=new long[n+1];}
    void add(int i,long v){ for(i++; i<=n; i+=i&-i) bit[i]+=v; }
    long sum(int i){ long s=0; for(i++; i>0; i-=i&-i) s+=bit[i]; return s; }
  }
  public static int kth(Fenwick ft, long k){ int idx=0; long bitMask=1; while((bitMask<<1) <= ft.n) bitMask<<=1; for(long d=bitMask; d!=0; d>>=1){ int next=idx+(int)d; if(next<=ft.n && ft.bit[next] < k){ k -= ft.bit[next]; idx = next; } } return idx; }
}
```

Exercises
- Range add + range sum queries (lazy segtree)
- Autocomplete with Trie prefixes
- K-th smallest element in a stream (heap or order-statistics)



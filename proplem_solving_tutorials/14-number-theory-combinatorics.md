## 14 — Number Theory & Combinatorics

GCD, extended GCD
```java
public class GCD {
  public static int gcd(int a, int b){ return b==0?a:gcd(b,a%b); }
  public static long[] extGcd(long a, long b){ if(b==0) return new long[]{a,1,0}; long[] r=extGcd(b,a%b); return new long[]{r[0], r[2], r[1]- (a/b)*r[2]}; }
}
```

Modular exponentiation and inverse
```java
public class ModArith {
  public static long modPow(long a,long e,long mod){ long r=1%mod; a%=mod; while(e>0){ if((e&1)==1) r=r*a%mod; a=a*a%mod; e>>=1; } return r; }
  public static long modInvPrime(long a,long mod){ return modPow(a, mod-2, mod); }
}
```

Sieve of Eratosthenes and SPF
```java
import java.util.*;

public class Sieve {
  public static boolean[] primeSieve(int n){ boolean[] is = new boolean[n+1]; Arrays.fill(is,true); if(n>=0) is[0]=false; if(n>=1) is[1]=false; for(int p=2;p*p<=n;p++) if(is[p]) for(int x=p*p;x<=n;x+=p) is[x]=false; return is; }
  public static int[] smallestPrimeFactor(int n){ int[] spf=new int[n+1]; for(int i=2;i<=n;i++) if(spf[i]==0){ spf[i]=i; if((long)i*i<=n) for(long j=(long)i*i;j<=n;j+=i) if(spf[(int)j]==0) spf[(int)j]=i; } return spf; }
}
```

nCr mod prime with precomputation
```java
public class Combinatorics {
  private static final long MOD = 1_000_000_007L;
  static long[] fact, invFact;
  public static void init(int n){ fact=new long[n+1]; invFact=new long[n+1]; fact[0]=1; for(int i=1;i<=n;i++) fact[i]=fact[i-1]*i%MOD; invFact[n]=ModArith.modInvPrime(fact[n], MOD); for(int i=n;i>0;i--) invFact[i-1]=invFact[i]*i%MOD; }
  public static long nCr(int n,int r){ if(r<0||r>n) return 0; return (((fact[n]*invFact[r])%MOD)*invFact[n-r])%MOD; }
}
```

Euler Totient (phi)
```java
public class Totient {
  public static int phi(int n){ int res=n; for(int p=2;p*p<=n;p++){ if(n%p==0){ while(n%p==0) n/=p; res-=res/p; } } if(n>1) res-=res/n; return res; }
}
```

Exercises
- Factorization using SPF
- Count coprime pairs in a range (inclusion-exclusion)
- Number of lattice paths modulo a prime



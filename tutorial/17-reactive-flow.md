# Reactive Streams with java.util.concurrent.Flow

Java 9 introduced the `Flow` API for reactive streams and backpressure.

## SubmissionPublisher Example
```java
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;

public class ReactiveFlowDemo {
    public static void main(String[] args) throws Exception {
        try (SubmissionPublisher<String> publisher = new SubmissionPublisher<>()) {
            publisher.subscribe(new LoggingSubscriber<>(5));
            for (int i = 0; i < 20; i++) publisher.submit("msg-" + i);
            // allow async processing
            Thread.sleep(500);
        }
    }

    static class LoggingSubscriber<T> implements Flow.Subscriber<T> {
        private Flow.Subscription subscription; private final long batch;
        LoggingSubscriber(long batch) { this.batch = batch; }
        public void onSubscribe(Flow.Subscription subscription) {
            this.subscription = subscription; subscription.request(batch);
        }
        public void onNext(T item) { System.out.println("Received: " + item); subscription.request(1); }
        public void onError(Throwable throwable) { throwable.printStackTrace(); }
        public void onComplete() { System.out.println("Completed"); }
    }
}
```

Notes:
- `SubmissionPublisher` is a simple publisher; for production, consider libraries like Project Reactor or RxJava for richer operators.



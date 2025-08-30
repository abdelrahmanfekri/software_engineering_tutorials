## 07 — Linked Lists, Trees, BST

Linked list essentials
- Slow/fast pointers for cycles and middle
- In-place reverse, merge two lists, remove Nth from end

Reverse a list
```java
public class ReverseList {
  static class ListNode { int val; ListNode next; ListNode(int v){val=v;} }
  public static ListNode reverse(ListNode head) {
    ListNode prev = null, current = head;
    while (current != null) {
      ListNode next = current.next;
      current.next = prev;
      prev = current;
      current = next;
    }
    return prev;
  }
}
```

Trees & BST
- Traversals: inorder, preorder, postorder; BFS level-order
- BST invariant: left < root < right (strict or non-strict by spec)

Validate BST (inorder)
```java
public class ValidateBST {
  static class TreeNode { int val; TreeNode left, right; TreeNode(int v){val=v;} }
  public static boolean isValidBST(TreeNode root) {
    return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
  }
  private static boolean validate(TreeNode node, long min, long max) {
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return validate(node.left, min, node.val) && validate(node.right, node.val, max);
  }
}
```

Exercises
- Merge two sorted lists; detect and remove cycle
- Binary tree level order traversal; zigzag level order
- Kth smallest in BST (inorder)



package bupt;

import java.util.LinkedList;
import java.util.Queue;

public class MyStack {
    Queue<Integer> queue1;//主要队列
    Queue<Integer> queue2;//辅助队列

    public MyStack(){
        queue1 = new LinkedList<>();//linkedlist可以当做队列使用的双向链表
        queue2 = new LinkedList<>();
    }
    public void push(int x){
        queue2.offer(x);//插入链表尾部
                        //先放在辅助队列
        while(!queue1.isEmpty()){
            queue2.offer(queue1.poll());//poll删除并返回链表头元素
                                        //弹出队列头
        }
        Queue<Integer> queueTemp;
        queueTemp = queue1;
        queue1 = queue2;
        queue2 = queueTemp;//交换队列12,将元素都放在队列1
    }
    public int pop(){
        return queue1.poll();
    }
    public int top(){
        return queue1.peek();//返回第一个元素
    }

    public boolean empty(){
        return queue1.isEmpty();
    }
}

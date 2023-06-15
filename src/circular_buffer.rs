use std::collections::VecDeque;

pub struct CircularBuffer<T> {
    deque: VecDeque<T>,
    cap: usize,
}
#[derive(Debug, PartialEq)]
pub enum CircularError {
    Empty,
    Full,
}
impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            deque: VecDeque::new(),
            cap: capacity,
        }
    }
    pub fn write(&mut self, element: T) -> Result<(), CircularError> {
        if self.deque.len() == self.cap {
            Err(CircularError::Full)
        } else {
            self.deque.push_back(element);
            Ok(())
        }
    }
    pub fn overwrite(&mut self, element: T) {
        if self.deque.len() == self.cap {
            let _ = self.deque.pop_front();
        }
        self.deque.push_back(element)
    }

    pub fn read(&mut self) -> Result<T, CircularError> {
        self.deque.pop_front().ok_or(CircularError::Empty)
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.deque.iter()
    }
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.deque.as_slices()
    }
    pub fn make_contiguous(&mut self) {
        self.deque.make_contiguous();
    }
    pub fn len(&self) -> usize {
        self.deque.len()
    }

    pub fn clear(&mut self) {
        self.deque = VecDeque::new();
    }
}

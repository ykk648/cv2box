
### CV Queue

A queue-like high-level class which can be used for two different python projects on same host machine.

Based on ShareMemory , which is more efficient than socket or grpc.

#### idea

![CVQueue](./src/cvqueue.png)

#### fixed problems

- Deal with memory release;
- Sync without lock;
- Multi size buffer trans;
- Add queue-like api like full/empty/close;
- Mem access delay to avoid access fail;
- More retry logic;

### CV Image

Human-like API to unite different image format, convenient way to convert/show/save one pic or even pic path.

Makes `BGR RGB PIL cv2 sk-image bytes base64 tensor` to be one **CVImage**.

#### example
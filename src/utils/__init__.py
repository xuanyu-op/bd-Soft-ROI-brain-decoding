'''
safe_get_source 这个实现的作用是确保整个 batch 的 subject 都相同；
如果相同，就返回其中那个唯一值；如果不相同，就直接 assert 爆掉。它当然“能得到东西”，
但前提是你传给它的是一个序列（列表/元组等），而且里面所有元素都相同；如果你直接传一个字符串（比如 "subject_3"），
这个实现会把字符串当作字符序列来处理，set("subject_3") 会包含很多不同字符，assert 会失败。
'''
'''
标识是否全部相同。
assert len(...) == 1：若不相同就直接中断（调试期的硬检查）。
return sources[0]：既然都相同，返回第一个即可（作为“唯一 subject”）。
关键点：这里的 sources 语义是“一个可迭代的集合/序列的 subject 标识”，例如：
["subject_3", "subject_3", "subject_3"]
("subject_1",)
这两种都会通过断言并返回 "subject_3" / "subject_1"。
'''
def safe_get_source(sources):
    assert len(set(sources)) == 1
    return sources[0]


def adapt_voxels(voxels, training=False):
    if training:
        if voxels.dim() == 3 or voxels.dim() == 5:  # mindeye's dataloader
            repeat_index = random.randint(0, 2)
            voxels = voxels[:, repeat_index]
    else:
        if voxels.dim() == 3 or voxels.dim() == 5:
            voxels = voxels.mean(dim=1)
    return voxels

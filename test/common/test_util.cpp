#include "gtest/gtest.h"
#include <set>

extern "C" {
#include <util.h>
}

TEST(read_file, success) {
    uint8_t* addr = read_file("test_file.txt");
    EXPECT_NE(nullptr, addr);

    EXPECT_EQ('T', addr[0]);
    EXPECT_EQ('E', addr[1]);
    EXPECT_EQ('S', addr[2]);
    EXPECT_EQ('T', addr[3]);

    free(addr);
}

TEST(read_file, error) {
    uint8_t* addr = read_file("foo.txt");
    EXPECT_EQ(nullptr, addr);
}

TEST(read_uint8, success) {
    uint8_t* dat = (uint8_t*)malloc(sizeof(uint8_t) * 3);
    dat[0] = 0;
    dat[1] = 1;
    dat[2] = 2;

    int pos = 0;
    EXPECT_EQ(0, read_uint8(dat, &pos));
    EXPECT_EQ(1, read_uint8(dat, &pos));
    EXPECT_EQ(2, read_uint8(dat, &pos));

    free(dat);
}

TEST(read_int32, success) {
    uint8_t dat[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2};
    uint8_t* u_dat = (uint8_t*)(&dat);
    int pos = 0;
    EXPECT_EQ(0, read_int32(u_dat, &pos));
    EXPECT_EQ(1, read_int32(u_dat, &pos));
    EXPECT_EQ(2, read_int32(u_dat, &pos));
}

TEST(choice, success) {
    int* nums = choice(100, 98);

    std::set<int> s;
    for (int i = 0; i < 98; ++i) {
        EXPECT_LE(0, nums[i]);
        EXPECT_LE(nums[i], 99);
        
        EXPECT_EQ(0, s.count(nums[i]));
        s.insert(nums[i]);
    }

    free(nums);
}
